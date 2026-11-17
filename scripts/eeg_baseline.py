"""
eeg_baseline.py - EEG-only baseline with leave-one-CHILD-out validation.

    python eeg_baseline.py --data eeg_windows.npz

Input : eeg_windows.npz from prepare_eeg.py  (X windows x 12 features, y 1=ASD/0=control, subject ids)
Needs : numpy, scikit-learn.  If importing them fails with "Application Control policy has blocked",
        run this on Google Colab instead (upload eeg_windows.npz + this file, then  !python eeg_baseline.py).

What it does
  * Leave-one-child-out: every child is predicted by a model that never saw ANY window of that child.
  * Two fixed models (no tuning, so no optimistic bias): logistic regression and random forest.
  * Reports window-level accuracy AND child-level accuracy (mean predicted probability over the child's windows).
    Child-level is the honest number - windows from one child are highly correlated.
  * Permutation test: shuffles which children are ASD and repeats the whole procedure, to show how often
    chance alone reaches your accuracy (with ~26 children this matters).
  * Saves eeg_oof.npz (out-of-fold probability for every window) - used later for the face+EEG fusion.
Writes eeg_baseline_report.txt and eeg_loso_children.csv.
"""
import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


def make_rf(seed=0):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=seed)


def loso(make_model, X, y, subj):
    """Return out-of-fold P(ASD) for every window."""
    oof = np.zeros(len(y))
    for s in np.unique(subj):
        te = subj == s
        m = make_model()
        m.fit(X[~te], y[~te])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def child_table(oof, y, subj):
    ids = np.unique(subj)
    p = np.array([oof[subj == s].mean() for s in ids])
    lab = np.array([y[subj == s][0] for s in ids])
    return ids, p, lab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="eeg_windows.npz")
    ap.add_argument("--perms", type=int, default=200, help="label permutations for the chance test (0 = skip)")
    a = ap.parse_args()

    d = np.load(a.data, allow_pickle=True)
    X, y, subj = d["X"], d["y"].astype(int), d["subject"].astype(str)
    ids = np.unique(subj)
    n_asd = int(sum(y[subj == s][0] == 1 for s in ids))
    n_ctl = len(ids) - n_asd
    buf = io.StringIO()

    def out(s=""):
        print(s)
        buf.write(s + "\n")

    out(f"Windows: {len(y)} | children: {len(ids)} (ASD {n_asd}, control {n_ctl}) | features: {X.shape[1]}")
    out(f"Chance level for child-level accuracy (always guess the bigger group): {max(n_asd, n_ctl) / len(ids):.3f}\n")

    results, oofs = {}, {}
    for name, mk in [("LogReg", make_lr), ("RandomForest", make_rf)]:
        oof = loso(mk, X, y, subj)
        oofs[name] = oof
        cid, cp, clab = child_table(oof, y, subj)
        cpred = (cp > 0.5).astype(int)
        win_acc = ((oof > 0.5).astype(int) == y).mean()
        ch_acc = (cpred == clab).mean()
        bal = balanced_accuracy_score(clab, cpred)
        auc = roc_auc_score(clab, cp)
        tn, fp, fn, tp = confusion_matrix(clab, cpred, labels=[0, 1]).ravel()
        results[name] = dict(ch_acc=ch_acc)
        out(f"== {name} ==")
        out(f"  window-level accuracy : {win_acc:.3f}   (inflated: neighbouring windows are correlated)")
        out(f"  child-level accuracy  : {ch_acc:.3f}   ({int((cpred == clab).sum())}/{len(cid)} children)")
        out(f"  child-level balanced accuracy: {bal:.3f} | AUC: {auc:.3f}")
        out(f"  confusion (children): TN={tn} FP={fp} FN={fn} TP={tp}   [sensitivity {tp / max(tp + fn, 1):.2f}, specificity {tn / max(tn + fp, 1):.2f}]\n")

    if a.perms > 0:
        rng = np.random.RandomState(0)
        out(f"== Chance test: shuffling which children are ASD, {a.perms} times (LogReg, child-level accuracy) ==")
        obs = results["LogReg"]["ch_acc"]
        lab_by_child = {s: y[subj == s][0] for s in ids}
        perm_acc = []
        for _ in range(a.perms):
            shuffled = dict(zip(ids, rng.permutation([lab_by_child[s] for s in ids])))
            yp = np.array([shuffled[s] for s in subj])
            oof = loso(make_lr, X, yp, subj)
            cid, cp, clab = child_table(oof, yp, subj)
            perm_acc.append(((cp > 0.5).astype(int) == clab).mean())
        perm_acc = np.array(perm_acc)
        pval = (1 + (perm_acc >= obs).sum()) / (1 + len(perm_acc))
        out(f"  shuffled-label accuracy: mean {perm_acc.mean():.3f}, 95th percentile {np.percentile(perm_acc, 95):.3f}")
        out(f"  observed {obs:.3f}  ->  p = {pval:.3f}  (chance alone matches or beats it in {pval * 100:.1f}% of shuffles)\n")

    # per-child table + OOF export for the fusion step
    cid, cp_lr, clab = child_table(oofs["LogReg"], y, subj)
    _, cp_rf, _ = child_table(oofs["RandomForest"], y, subj)
    out("== Per child (mean predicted P(ASD) over its windows) ==")
    out(f"  {'child':9s} {'true':>5s} {'LogReg':>7s} {'RF':>6s}")
    with open("eeg_loso_children.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["child", "true_class", "p_asd_logreg", "p_asd_rf"])
        for s, l, p1, p2 in zip(cid, clab, cp_lr, cp_rf):
            mark = "" if ((p1 > 0.5) == bool(l)) else "  <- wrong (LogReg)"
            out(f"  {s:9s} {'ASD' if l else 'CTRL':>5s} {p1:7.2f} {p2:6.2f}{mark}")
            w.writerow([s, int(l), f"{p1:.4f}", f"{p2:.4f}"])
    np.savez("eeg_oof.npz", y=y, subject=subj, p_logreg=oofs["LogReg"], p_rf=oofs["RandomForest"])
    out("\nSaved eeg_oof.npz, eeg_loso_children.csv, eeg_baseline_report.txt")
    Path("eeg_baseline_report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
