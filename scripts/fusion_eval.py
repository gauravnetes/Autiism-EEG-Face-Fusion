"""
fusion_eval.py - simulated fusion of face + EEG, with ablation table.

IMPORTANT (state this in the report): the two datasets contain DIFFERENT people. A "pair" is a
(face image, EEG window) of the SAME CLASS, as in the reference paper. Fused accuracy therefore
benefits partly by construction (two independent votes for the same class).

Protocol (leave-one-child-out for EEG, fixed face test split), repeated R times with new random pairings:
  for each held-out EEG child c (class k):
      TEST  pairs  = every clean TEST face of class k  x  one random window of child c
      TRAIN pairs  = every TRAIN face  x  a random same-class window from the OTHER children
  EEG-only     : out-of-fold P(ASD) of the paired window (model never saw child c)
  Face-only    : fine-tuned CNN P(ASD) of the paired face
  Late fusion  : (p_face + p_eeg) / 2 > 0.5               (no training)
  Feature fusion: LogReg on [PCA-32 of frozen ImageNet face embedding | EEG window features], trained on TRAIN pairs
  Mismatched control: face of the OTHER class + EEG window of child c -> who wins the fused decision?

Usage:
  python fusion_eval.py --eeg eeg_windows_v2.npz --variant A --face_out face_out_rgb.npz --face_emb face_emb.npz
"""
import argparse, warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import binary_metrics, wilson_ci

warnings.filterwarnings("ignore")
VARIANTS = {"A": ("base",), "B": ("base", "dwt"), "C": ("base", "conn"), "D": ("base", "dwt", "conn"), "E": ("base", "dwt", "conn", "spec")}


def eeg_clf():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


def ms(v):
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg", default="eeg_windows_v2.npz")
    ap.add_argument("--variant", default="A", choices=list(VARIANTS))
    ap.add_argument("--face_out", default="face_out_rgb.npz")
    ap.add_argument("--face_emb", default="face_emb.npz")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--pca", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    # ---------- EEG: out-of-fold window probabilities ----------
    d = np.load(a.eeg, allow_pickle=True)
    ids, ylab, groups = list(d["ids"]), d["labels"].astype(int), d["groups"]
    cols = np.isin(groups, VARIANTS[a.variant])
    X = d["X"][:, cols]
    cidx = np.array([ids.index(c) for c in d["child"]])
    nC = len(ids)
    ywin = ylab[cidx]
    p_eeg = np.zeros(len(X))
    Z, tr_cls = [], []
    for c in range(nC):
        te = cidx == c
        m = eeg_clf().fit(X[~te], ywin[~te])
        p_eeg[te] = m.predict_proba(X[te])[:, 1]
        sc = StandardScaler().fit(X[~te])
        Z.append(sc.transform(X))
        tr = np.flatnonzero(~te)
        tr_cls.append({0: tr[ywin[tr] == 0], 1: tr[ywin[tr] == 1]})
    child_p = np.bincount(cidx, weights=p_eeg) / np.bincount(cidx)
    eeg_child = binary_metrics(ylab, child_p)

    # ---------- Face ----------
    fo, fe = np.load(a.face_out, allow_pickle=True), np.load(a.face_emb, allow_pickle=True)
    p_face, y_face = fo["test_prob"], fo["test_y"].astype(int)
    assert np.array_equal(y_face, fe["test_y"].astype(int)), "face_out and face_emb test order differ - rerun face_cnn.py"
    s1 = StandardScaler().fit(fe["train_emb"])
    pca = PCA(a.pca, random_state=0).fit(s1.transform(fe["train_emb"]))
    Ftr, Fte = pca.transform(s1.transform(fe["train_emb"])), pca.transform(s1.transform(fe["test_emb"]))
    s2 = StandardScaler().fit(Ftr)
    Ftr, Fte = s2.transform(Ftr), s2.transform(Fte)
    ytr = fe["train_y"].astype(int)
    face_only_img = binary_metrics(y_face, p_face)

    # ---------- Repeated random pairings ----------
    names = ["EEG-only (window)", "Face-only (CNN)", "Late fusion (avg)", "Feature-level fusion"]
    R = {n: [] for n in names}
    mis = {"Late fusion (avg)": [], "Feature-level fusion": []}
    for r in range(a.reps):
        rng = np.random.default_rng(a.seed + r)
        Y, PE, PF, PL, PX = [], [], [], [], []
        mm = {k: [0, 0, 0] for k in mis}               # agree-with-face, agree-with-EEG, n
        for c in range(nC):
            k = int(ylab[c])
            wsel = np.empty(len(ytr), dtype=int)
            for cl in (0, 1):
                m_ = ytr == cl
                wsel[m_] = rng.choice(tr_cls[c][cl], m_.sum())
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(np.hstack([Ftr, Z[c][wsel]]), ytr)
            w = rng.choice(np.flatnonzero(cidx == c))
            same, other = np.flatnonzero(y_face == k), np.flatnonzero(y_face != k)
            ez = np.tile(Z[c][w], (len(same), 1))
            px = clf.predict_proba(np.hstack([Fte[same], ez]))[:, 1]
            Y.append(np.full(len(same), k)); PE.append(np.full(len(same), p_eeg[w])); PF.append(p_face[same])
            PL.append((p_face[same] + p_eeg[w]) / 2); PX.append(px)
            # mismatched control: face of the other class, EEG of child c (class k)
            pxm = clf.predict_proba(np.hstack([Fte[other], np.tile(Z[c][w], (len(other), 1))]))[:, 1]
            plm = (p_face[other] + p_eeg[w]) / 2
            for nm, pr in (("Late fusion (avg)", plm), ("Feature-level fusion", pxm)):
                pred = (pr > 0.5).astype(int)
                mm[nm][0] += int((pred == 1 - k).sum()); mm[nm][1] += int((pred == k).sum()); mm[nm][2] += len(pred)
        Y, PE, PF, PL, PX = map(np.concatenate, (Y, PE, PF, PL, PX))
        for nm, p in zip(names, (PE, PF, PL, PX)):
            R[nm].append(binary_metrics(Y, p))
        for nm in mis:
            mis[nm].append((mm[nm][0] / mm[nm][2], mm[nm][1] / mm[nm][2]))

    # ---------- Report ----------
    L = [f"Fusion evaluation | EEG variant {a.variant} ({int(cols.sum())} feats) | {nC} EEG children | {len(y_face)} test faces | {a.reps} random pairings",
         f"Test pairs per repetition: {sum(int((y_face == ylab[c]).sum()) for c in range(nC))}", "",
         f"| {'Model':<24}| Accuracy        | Balanced acc.   | Sensitivity     | Specificity     | AUC             |",
         "|" + "-" * 25 + ("|" + "-" * 17) * 5 + "|"]
    for nm in names:
        g = lambda key: ms([x[key] for x in R[nm]])
        L.append(f"| {nm:<24}| {g('acc'):<16}| {g('bal'):<16}| {g('sens'):<16}| {g('spec'):<16}| {g('auc'):<16}|")
    lo, hi = wilson_ci(face_only_img["k"], face_only_img["n"])
    L += ["", f"EEG-only, child level (LOCO, mean P(ASD) over windows): acc {eeg_child['acc']:.3f} ({eeg_child['k']}/{nC}) | bal {eeg_child['bal']:.3f} | AUC {eeg_child['auc']:.3f}",
          f"Face-only on the {face_only_img['n']} test images (unweighted): acc {face_only_img['acc']:.3f} (95% CI {lo:.3f}-{hi:.3f}) | AUC {face_only_img['auc']:.3f}",
          "(Face-only in the table is weighted by pairing: each ASD face appears once per ASD child, each control face once per control child.)",
          "", "Mismatched-pair control (face of one class + EEG of the other): which modality wins the fused decision?"]
    for nm, v in mis.items():
        v = np.array(v)
        L.append(f"  {nm:<22} follows FACE label {v[:, 0].mean():.3f} | follows EEG label {v[:, 1].mean():.3f}")
    L += ["", "Read-me for the report: pairs are class-matched across different people, so fusion gains are partly by construction.",
          "Real uncertainty is dominated by the small number of EEG children (26) and test faces (194); ± shows only pairing randomness."]
    rep = "\n".join(L)
    print(rep)
    tag = a.tag or a.variant
    open(f"fusion_report_{tag}.txt", "w").write(rep + "\n")


if __name__ == "__main__":
    main()
