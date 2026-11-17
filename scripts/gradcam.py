"""
gradcam.py - where does the CNN look? Saves gradcam_<variant>.png.

If heat sits on the face, good. If it sits on background, borders, hair or corners, the model is using shortcuts.

Usage: python gradcam.py --root /content/faces --manifest faces_clean_manifest.csv --variant rgb
"""
import argparse
import numpy as np
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from face_cnn import make_tf, new_backbone, gray_eq
from common import list_faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--variant", default="rgb", choices=["rgb", "grayeq"])
    ap.add_argument("--per_group", type=int, default=3)
    a = ap.parse_args()

    data = list_faces(a.root, a.manifest)
    paths, y = data["test"]
    out = np.load(f"face_out_{a.variant}.npz", allow_pickle=True)
    prob = out["test_prob"]
    assert np.array_equal(out["test_y"].astype(int), y), "test ordering differs from face_out file"

    model = new_backbone(False)
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(torch.load(f"face_resnet18_{a.variant}.pt", map_location="cpu"))
    model.eval()
    acts, grads = {}, {}
    def fwd(m, i, o):                                # tensor-level hook: avoids the in-place ReLU / full-backward-hook error
        acts["a"] = o
        o.register_hook(lambda g: grads.__setitem__("g", g))
    model.layer4.register_forward_hook(fwd)
    tf = make_tf(a.variant, False)

    pred = (prob > 0.5).astype(int)
    groups = [("ASD, correct", (y == 1) & (pred == 1)), ("Control, correct", (y == 0) & (pred == 0)),
              ("Wrong predictions", pred != y)]
    picks = []
    for title, m in groups:
        idx = np.flatnonzero(m)
        sel = idx[np.linspace(0, len(idx) - 1, min(a.per_group, len(idx))).astype(int)] if len(idx) else []
        picks += [(title, int(i)) for i in sel]

    fig, ax = plt.subplots(2, len(picks), figsize=(2.4 * len(picks), 5.2))
    ax = np.atleast_2d(ax).reshape(2, -1)
    for j, (title, i) in enumerate(picks):
        im = Image.open(paths[i]).convert("RGB")
        x = tf(im).unsqueeze(0).requires_grad_(True)
        model.zero_grad()
        model(x).squeeze().backward()
        w = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts["a"]).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        cam = (cam / (cam.max() + 1e-8)).detach().numpy()
        base = np.asarray((gray_eq(im) if a.variant == "grayeq" else im).resize((224, 224)))
        ax[0, j].imshow(base); ax[0, j].axis("off")
        ax[0, j].set_title(f"{title}\ntrue {'ASD' if y[i] else 'CTRL'}  P(ASD)={prob[i]:.2f}", fontsize=7)
        ax[1, j].imshow(base); ax[1, j].imshow(cam, cmap="jet", alpha=0.45); ax[1, j].axis("off")
    plt.tight_layout()
    plt.savefig(f"gradcam_{a.variant}.png", dpi=150)
    print(f"saved gradcam_{a.variant}.png")


if __name__ == "__main__":
    main()
