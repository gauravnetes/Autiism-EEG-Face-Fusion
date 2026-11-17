"""
face_shortcut_test.py - can photo properties ALONE (no face content) predict ASD vs control?

If yes, a CNN can cheat the same way. Runs two feature sets:
  RAW  : width, height, aspect, mean brightness, file size, mean R/G/B   (before preprocessing)
  POST : brightness, contrast, mean R/G/B measured after resize to 224x224 (what the CNN actually sees)
Models: LogReg and RandomForest (RF catches rules like "height == 224" that LogReg cannot).
Train on train split, report on valid and test with 95% Wilson CI.

Usage: python face_shortcut_test.py --root ./Autistic_Children_Facial_Image --manifest faces_clean_manifest.csv
"""
import argparse, os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import list_faces, binary_metrics, wilson_ci


def raw_feats(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    a = np.asarray(im, dtype=np.float32)
    return [w, h, w / h, a.mean(), os.path.getsize(path) / 1024.0, *a.reshape(-1, 3).mean(0)]


def post_feats(path):
    im = Image.open(path).convert("RGB").resize((224, 224))
    a = np.asarray(im, dtype=np.float32)
    return [a.mean(), a.std(), *a.reshape(-1, 3).mean(0)]


def build(data, fn):
    return {sp: (np.array([fn(p) for p in paths]), y) for sp, (paths, y) in data.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default="face_shortcut_report.txt")
    a = ap.parse_args()

    data = list_faces(a.root, a.manifest)
    lines = []
    n_test = len(data["test"][1])
    upper = 0.5 + 1.96 * np.sqrt(0.25 / n_test)
    lines.append(f"Test images: {n_test}. If photo properties carry no label information, accuracy should stay below ~{upper:.3f}.")
    lines.append("")
    for name, fn in (("RAW (before preprocessing)", raw_feats), ("POST (after 224x224 resize)", post_feats)):
        F = build(data, fn)
        Xtr, ytr = F["train"]
        lines.append(f"== {name}: {Xtr.shape[1]} features ==")
        for mname, mk in (("LogReg", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                          ("RandomForest", lambda: RandomForestClassifier(300, max_depth=6, random_state=0))):
            m = mk().fit(Xtr, ytr)
            for sp in ("valid", "test"):
                X, y = F[sp]
                r = binary_metrics(y, m.predict_proba(X)[:, 1])
                lo, hi = wilson_ci(r["k"], r["n"])
                flag = "  <-- LEAK: photo properties alone beat chance" if sp == "test" and lo > 0.5 else ""
                lines.append(f"  {mname:<13}{sp:<6} acc {r['acc']:.3f}  (95% CI {lo:.3f}-{hi:.3f})  AUC {r['auc']:.3f}  bal {r['bal']:.3f}{flag}")
        lines.append("")
    lines += ["How to read: a TEST accuracy whose CI excludes 0.5 means the classes differ in photo style.",
              "Then the CNN number is not purely about faces. Report this number next to the CNN accuracy and run the",
              "grayscale + histogram-equalised robustness variant (face_cnn.py --variant grayeq)."]
    rep = "\n".join(lines)
    print(rep)
    open(a.out, "w").write(rep + "\n")


if __name__ == "__main__":
    main()
