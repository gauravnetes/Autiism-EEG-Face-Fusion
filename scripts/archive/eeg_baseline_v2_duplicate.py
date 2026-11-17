"""
eeg_baseline_v2.py - leave-one-child-out comparison of pre-declared feature variants.

Variants (declared up front, all reported):
  A = base                      <- PRIMARY (what v1 used)
  B = base + dwt
  C = base + conn
  D = base + dwt + conn
  E = D + spec                  (only if prepare_eeg_v2.py was run with --fs)
Models: LogReg and RBF-SVM, fixed settings, no tuning.

Permutation test: labels of children are shuffled; ALL variant x model combos are re-run on each
shuffle and the BEST accuracy is recorded. The p-value for "best combo" therefore accounts for
having tried many combos (selection-corrected).

Usage: python eeg_baseline_v2.py --data eeg_windows_v2.npz --n_perm 100
"""
import argparse, warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


def make_model(kind):
    clf = (LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000) if kind == "LogReg"
           else SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))
    return make_pipeline(StandardScaler(), clf)


def loco_child_scores(X, cidx, ylab, kind):
    """Mean decision score per child from a model that never saw that child."""
    n = len(ylab)
    win_score = np.zeros(len(X))
    ywin = ylab[cidx]
    for c in range(n):
        te = cidx == c
        m = make_model(kind).fit(X[~te], ywin[~te])
        win_score[te] = m.decision_function(X[te])
    child = np.bincount(cidx, weights=win_score, minlength=n) / np.bincount(cidx, minlength=n)
    return child, win_score


def child_metrics(score, ylab):
    pred = (score > 0).astype(int)
    tp = int(((pred == 1) & (ylab == 1)).sum()); tn = int(((pred == 0) & (ylab == 0)).sum())
    fp = int(((pred == 1) & (ylab == 0)).sum()); fn = int(((pred == 0) & (ylab == 1)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    auc = roc_auc_score(ylab, score) if len(set(ylab)) > 1 else float("nan")
    return dict(acc=(tp + tn) / len(ylab), k=tp + tn, bal=(sens + spec) / 2, auc=auc,
                sens=sens, spec=spec, tn=tn, fp=fp, fn=fn, tp=tp)


def one_shuffle(Xs, cidx, perm, combos):
    out = []
    for X, kind in combos:
        s, _ = loco_child_scores(X, cidx, perm, kind)
        out.append(((s > 0).astype(int) == perm).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eeg_windows_v2.npz")
    ap.add_argument("--n_perm", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = np.load(a.data, allow_pickle=True)
    X, child, ids, ylab = d["X"], d["child"], list(d["ids"]), d["labels"].astype(int)
    groups = d["groups"]
    cidx = np.array([ids.index(c) for c in child])
    n = len(ids)
    chance = max(ylab.mean(), 1 - ylab.mean())

    def cols(*g):
        return np.isin(groups, g)
    variants = {"A base": cols("base"), "B base+dwt": cols("base", "dwt"),
                "C base+conn": cols("base", "conn"), "D base+dwt+conn": cols("base", "dwt", "conn")}
    if "spec" in groups:
        variants["E D+spec"] = cols("base", "dwt", "conn", "spec")

    combos, labels = [], []
    for vn, m in variants.items():
        for kind in ("LogReg", "SVM"):
            combos.append((X[:, m], kind)); labels.append((vn, kind, int(m.sum())))

    lines = [f"Windows: {len(X)} | children: {n} (ASD {ylab.sum()}, control {n - ylab.sum()})",
             f"Chance level for child-level accuracy: {chance:.3f}",
             "Primary (pre-declared): A base + LogReg. Everything else is exploratory.", ""]

    obs, scores = [], []
    for (Xv, kind), lb in zip(combos, labels):
        s, _ = loco_child_scores(Xv, cidx, ylab, kind)
        scores.append(s); obs.append(child_metrics(s, ylab))

    rng = np.random.default_rng(a.seed)
    perms = [rng.permutation(ylab) for _ in range(a.n_perm)]
    null = np.array(Parallel(n_jobs=a.jobs)(delayed(one_shuffle)(None, cidx, p, combos) for p in perms))  # (n_perm, n_combo)
    null_max = null.max(1)

    lines.append(f"{'variant':<17}{'model':<8}{'#f':>4}{'child acc':>11}{'bal acc':>9}{'AUC':>7}{'sens':>6}{'spec':>6}{'p(uncorr)':>11}")
    for i, ((vn, kind, nf), o) in enumerate(zip(labels, obs)):
        p_unc = (np.sum(null[:, i] >= o["acc"] - 1e-9) + 1) / (a.n_perm + 1)
        lines.append(f"{vn:<17}{kind:<8}{nf:>4}{o['acc']:>7.3f} ({o['k']:>2}/{n}){o['bal']:>8.3f}{o['auc']:>7.3f}"
                     f"{o['sens']:>6.2f}{o['spec']:>6.2f}{p_unc:>11.3f}")
    best = int(np.argmax([o["acc"] for o in obs]))
    p_best = (np.sum(null_max >= obs[best]["acc"] - 1e-9) + 1) / (a.n_perm + 1)
    lines += ["",
              f"Permutation test, {a.n_perm} shuffles of which children are ASD, ALL {len(combos)} combos re-run each time:",
              f"  best-of-all-combos accuracy under shuffling: mean {null_max.mean():.3f}, 95th percentile {np.percentile(null_max, 95):.3f}",
              f"  best observed combo = {labels[best][0]} + {labels[best][1]}: {obs[best]['acc']:.3f}  -> selection-corrected p = {p_best:.3f}",
              "  (p(uncorr) in the table ignores that many combos were tried; use the corrected p for the best one.)",
              "", "Per child (mean decision score over its windows; >0 means predicted ASD)",
              f"  {'child':<9}{'true':<6}{'A+LogReg':>10}{'best':>10}"]
    pa = labels.index(("A base", "LogReg", int(variants["A base"].sum())))
    for j, cid in enumerate(ids):
        mark = "" if (scores[pa][j] > 0) == bool(ylab[j]) else "  <- wrong (A+LogReg)"
        lines.append(f"  {cid:<9}{'ASD' if ylab[j] else 'CTRL':<6}{scores[pa][j]:>10.2f}{scores[best][j]:>10.2f}{mark}")
    rep = "\n".join(lines)
    print(rep)
    open("eeg_v2_report.txt", "w").write(rep + "\n")
    np.savez("eeg_v2_oof.npz", child_scores=np.array(scores), labels=ylab, ids=np.array(ids),
             combos=np.array([f"{v}|{k}" for v, k, _ in labels]), win_child=child)
    print("\nSaved eeg_v2_report.txt, eeg_v2_oof.npz")


if __name__ == "__main__":
    main()
