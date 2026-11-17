"""
face_cnn.py - ResNet18 face classifier (ASD vs control), two-stage fine-tuning.

Stage 1: train the new head only (frozen backbone).  Stage 2: also unfreeze layer4 at a lower LR.
Early stopping / model selection on VALID loss. TEST is evaluated exactly once, at the end.

Also exports FROZEN ImageNet 512-d embeddings (face_emb.npz) for feature-level fusion.
We never export predictions or embeddings of the fine-tuned net for TRAIN images (they are overfit).

Usage (Colab GPU):
  python face_cnn.py --root /content/faces --manifest faces_clean_manifest.csv --variant rgb
  python face_cnn.py --root /content/faces --manifest faces_clean_manifest.csv --variant grayeq   # robustness check
"""
import argparse, copy, random, time
import numpy as np
import torch, torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
from common import list_faces, binary_metrics, wilson_ci

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def gray_eq(im):
    return ImageOps.equalize(ImageOps.grayscale(im)).convert("RGB")


def make_tf(variant, train):
    pre = [T.Lambda(gray_eq)] if variant == "grayeq" else []
    aug = [T.RandomHorizontalFlip(), T.RandomRotation(10), T.ColorJitter(brightness=0.3, contrast=0.3)] if train else []
    return T.Compose(pre + [T.Resize((224, 224))] + aug + [T.ToTensor(), T.Normalize(MEAN, STD)])


class Faces(Dataset):
    def __init__(self, paths, y, tf):
        self.paths, self.y, self.tf = paths, y, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), torch.tensor(float(self.y[i]))


def loader(paths, y, tf, bs, shuffle, workers):
    return DataLoader(Faces(paths, y, tf), batch_size=bs, shuffle=shuffle, num_workers=workers, pin_memory=True)


def new_backbone(pretrained):
    w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return models.resnet18(weights=w)


@torch.no_grad()
def embed(model, dl, dev):
    model.eval()
    fe = nn.Sequential(*list(model.children())[:-1])
    out = [fe(x.to(dev)).flatten(1).cpu().numpy() for x, _ in dl]
    return np.concatenate(out)


@torch.no_grad()
def predict(model, dl, dev):
    model.eval()
    probs, loss, n = [], 0.0, 0
    crit = nn.BCEWithLogitsLoss(reduction="sum")
    for x, y in dl:
        z = model(x.to(dev)).squeeze(1)
        loss += crit(z, y.to(dev)).item(); n += len(y)
        probs.append(torch.sigmoid(z).cpu().numpy())
    return np.concatenate(probs), loss / n


def set_mode(model, stage):
    model.train()
    if stage == 1:                                   # frozen backbone: keep BN statistics fixed
        for n_, m in model.named_children():
            if n_ != "fc":
                m.eval()
    else:                                            # only layer4 is trainable
        for n_, m in model.named_children():
            if n_ in ("conv1", "bn1", "layer1", "layer2", "layer3"):
                m.eval()


def run_stage(model, stage, epochs, lr, dl_tr, dl_va, dev, best, log, patience):
    for n_, p in model.named_parameters():
        p.requires_grad = n_.startswith("fc") or (stage == 2 and n_.startswith("layer4"))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    crit = nn.BCEWithLogitsLoss()
    bad = 0
    for ep in range(1, epochs + 1):
        set_mode(model, stage)
        t0, tl, nn_ = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(model(x).squeeze(1), y)
            loss.backward(); opt.step()
            tl += loss.item() * len(y); nn_ += len(y)
        pv, vl = predict(model, dl_va, dev)
        va = binary_metrics(dl_va.dataset.y, pv)["acc"]
        log.append(f"stage {stage} epoch {ep:>2}: train loss {tl / nn_:.4f} | valid loss {vl:.4f} acc {va:.3f} | {time.time() - t0:.0f}s")
        print(log[-1], flush=True)
        if vl < best["loss"] - 1e-4:
            best.update(loss=vl, state=copy.deepcopy(model.state_dict()), tag=f"stage {stage} epoch {ep}"); bad = 0
        else:
            bad += 1
            if stage == 2 and bad >= patience:
                log.append("early stopping"); print(log[-1]); break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--variant", choices=["rgb", "grayeq"], default="rgb")
    ap.add_argument("--epochs1", type=int, default=5)
    ap.add_argument("--epochs2", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no_pretrained", action="store_true", help="testing only")
    ap.add_argument("--max_per_split", type=int, default=0, help="debug: subsample")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    data = list_faces(a.root, a.manifest)
    if a.max_per_split:
        rng = np.random.default_rng(0)
        for sp in data:
            idx = np.sort(rng.choice(len(data[sp][1]), min(a.max_per_split, len(data[sp][1])), replace=False))
            data[sp] = ([data[sp][0][i] for i in idx], data[sp][1][idx])
    ev = make_tf(a.variant, False)
    dl = {sp: loader(*data[sp], ev, 64, False, a.workers) for sp in data}
    dl_train = loader(*data["train"], make_tf(a.variant, True), a.batch, True, a.workers)

    # frozen ImageNet embeddings (variant-independent; computed on the plain rgb pipeline)
    if a.variant == "rgb":
        base = new_backbone(not a.no_pretrained).to(dev)
        plain = make_tf("rgb", False)
        emb = {sp: embed(base, loader(*data[sp], plain, 64, False, a.workers), dev) for sp in data}
        np.savez("face_emb.npz", **{f"{sp}_emb": emb[sp] for sp in data}, **{f"{sp}_y": data[sp][1] for sp in data},
                 test_names=np.array([p.replace("\\", "/").split("/")[-1] for p in data["test"][0]]))
        print("saved face_emb.npz", {sp: emb[sp].shape for sp in emb})
        del base

    model = new_backbone(not a.no_pretrained)
    model.fc = nn.Linear(512, 1)
    model.to(dev)
    best, log = dict(loss=1e9, state=None, tag=""), []
    run_stage(model, 1, a.epochs1, 1e-3, dl_train, dl["valid"], dev, best, log, a.patience)
    run_stage(model, 2, a.epochs2, 1e-4, dl_train, dl["valid"], dev, best, log, a.patience)
    model.load_state_dict(best["state"])
    torch.save(best["state"], f"face_resnet18_{a.variant}.pt")

    pv, _ = predict(model, dl["valid"], dev)
    pt, _ = predict(model, dl["test"], dev)               # TEST: evaluated once
    rv, rt = binary_metrics(data["valid"][1], pv), binary_metrics(data["test"][1], pt)
    lo, hi = wilson_ci(rt["k"], rt["n"])
    names = np.array([p.replace("\\", "/").split("/")[-1] for p in data["test"][0]])
    np.savez(f"face_out_{a.variant}.npz", valid_prob=pv, valid_y=data["valid"][1], test_prob=pt, test_y=data["test"][1], test_names=names)
    rep = [f"Face CNN (ResNet18), variant={a.variant}, seed={a.seed}, selected {best['tag']}", *log, "",
           f"VALID acc {rv['acc']:.3f} | AUC {rv['auc']:.3f}",
           f"TEST  acc {rt['acc']:.3f} (95% CI {lo:.3f}-{hi:.3f}, n={rt['n']}) | balanced {rt['bal']:.3f} | AUC {rt['auc']:.3f}",
           f"TEST  sensitivity {rt['sens']:.3f} | specificity {rt['spec']:.3f} | confusion TN={rt['tn']} FP={rt['fp']} FN={rt['fn']} TP={rt['tp']}"]
    print("\n".join(rep[-3:]))
    open(f"face_cnn_report_{a.variant}.txt", "w").write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main()
