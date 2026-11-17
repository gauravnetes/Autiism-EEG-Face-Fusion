"""
eeg_audit.py - checks the Mendeley ASD-EEG-MCHAT dataset before we build on it.
Needs only numpy (no pandas), so it runs even where pandas is blocked.

    python eeg_audit.py --root .\\ASD-EEG-MCHAT-Dataset

Checks:
  1. Extracted Features: are any of the four feature files identical copies of each other?
     how extreme are the outliers (esp. Coefficient of Variation)?
  2. Raw EEG: per subject -> rows, channels stuck at the 24-bit ADC limit or flat,
     repeated rows (corrupt/looped data), and the signal scale (differs a lot between subjects).
  3. M-CHAT: per-subject item totals (just to see how they look; we will NOT use them as input).
Writes eeg_audit.txt - send me that file.
"""
import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np

RAIL_LO, RAIL_HI = -8388608, 8388607      # 24-bit ADC limits (open / saturated electrode)


def load_feature_csv(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r if row]
    names = [row[0] for row in rows]
    X = np.array([[float(v) for v in row[1:-1]] for row in rows])
    y = np.array([int(float(row[-1])) for row in rows])
    return header, names, X, y


def audit_features(root):
    print("=" * 70)
    print("1. EXTRACTED FEATURES")
    print("=" * 70)
    files = sorted(root.rglob("Extracted Features/*.csv"))
    data = {}
    for p in files:
        header, names, X, y = load_feature_csv(p)
        data[p.stem] = dict(names=names, X=X, y=y)
        ax = np.abs(X)
        print(f"\n{p.name}: {X.shape[0]} subjects x {X.shape[1]} channels | "
              f"class 1 (ASD)={int((y == 1).sum())}, class 0 (control)={int((y == 0).sum())}")
        print(f"  max |value| = {ax.max():,.1f}   99th pct = {np.percentile(ax, 99):,.1f}   "
              f"median = {np.median(ax):,.3f}   cells with |value| > 1000: {int((ax > 1000).sum())}")

    print("\n-- Are any feature files copies of each other? --")
    stems = list(data)
    for i in range(len(stems)):
        for j in range(i + 1, len(stems)):
            a, b = data[stems[i]], data[stems[j]]
            idx = {n: k for k, n in enumerate(b["names"])}
            if set(a["names"]) != set(b["names"]):
                print(f"  {stems[i]} vs {stems[j]}: different subject lists")
                continue
            order = [idx[n] for n in a["names"]]
            same = np.isclose(a["X"], b["X"][order], rtol=1e-9, atol=1e-12)
            frac = same.mean()
            flag = "  <-- IDENTICAL, one of them is a copy" if frac > 0.99 else ""
            print(f"  {stems[i]} vs {stems[j]}: {frac * 100:.1f}% of cells identical{flag}")


def audit_raw(root, min_rows):
    print("\n" + "=" * 70)
    print("2. RAW EEG RECORDINGS")
    print("=" * 70)
    files = sorted(root.rglob("*_Subject_*.csv"))
    if not files:
        print("No *_Subject_*.csv files found.")
        return
    print(f"{'subject':16s} {'rows':>7s} {'bad_ch':>6s} {'rail%':>6s} {'uniq_rows%':>10s} {'scale(MAD)':>12s}  notes")
    summary = {"ASD": [], "CTRL": []}
    for p in files:
        cls = "ASD" if p.name.upper().startswith("ASD") else "CTRL"
        a = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
        n, c = a.shape
        rail = ((a <= RAIL_LO) | (a >= RAIL_HI)).mean(axis=0)
        flat = a.std(axis=0) == 0
        bad = (rail > 0.5) | flat
        head = a[:20000]
        uniq = len({tuple(r) for r in head}) / len(head)
        good = a[:, ~bad] if (~bad).any() else a
        scale = float(np.median(np.median(np.abs(good - np.median(good, axis=0)), axis=0)))
        notes = []
        if n < min_rows:
            notes.append(f"TOO SHORT (<{min_rows})")
        if bad.sum() >= 4:
            notes.append("half or more channels unusable")
        elif bad.sum() > 0:
            notes.append(f"{int(bad.sum())} stuck/flat channel(s): {[int(k) + 1 for k in np.where(bad)[0]]}")
        if uniq < 0.5:
            notes.append("many repeated rows")
        print(f"{p.stem:16s} {n:7d} {int(bad.sum()):6d} {rail.mean() * 100:6.1f} {uniq * 100:10.1f} {scale:12.1f}  {'; '.join(notes)}")
        summary[cls].append(dict(name=p.stem, rows=n, bad=int(bad.sum()), ok=(n >= min_rows and bad.sum() < 4 and uniq >= 0.5)))

    print("\n-- Per class --")
    for cls, items in summary.items():
        if not items:
            continue
        rows = np.array([i["rows"] for i in items])
        usable = [i["name"] for i in items if i["ok"]]
        dropped = [i["name"] for i in items if not i["ok"]]
        print(f"  {cls}: {len(items)} subjects | rows min/median/max = {rows.min()}/{int(np.median(rows))}/{rows.max()} "
              f"| usable: {len(usable)} | questionable: {dropped}")
    print("\n  (scale differing by orders of magnitude between subjects means each recording must be "
          "normalised on its own; a model could otherwise learn the recording setup instead of ASD.)")


def audit_mchat(root):
    print("\n" + "=" * 70)
    print("3. M-CHAT (looked at only; NOT to be used as model input - it is a screening score for the same label)")
    print("=" * 70)
    for p in sorted(root.rglob("*M-CHAT.csv")):
        with p.open(encoding="utf-8-sig", newline="") as f:
            r = csv.reader(f)
            header = next(r)
            rows = [row for row in r if row]
        totals = {row[0]: int(sum(float(v) for v in row[1:])) for row in rows}
        vals = list(totals.values())
        print(f"\n{p.name}: {len(rows)} participants, {len(header) - 1} items | item-sum min/median/max = "
              f"{min(vals)}/{int(np.median(vals))}/{max(vals)}")
        print("  " + ", ".join(f"{k}={v}" for k, v in totals.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=r"..\data\raw\eeg\ASD-EEG-MCHAT-Dataset")
    ap.add_argument("--min-rows", type=int, default=1000, help="recordings shorter than this are flagged")
    a = ap.parse_args()
    root = Path(a.root)
    if not root.exists():
        sys.exit(f"Path not found: {root}")

    buf = io.StringIO()

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)

        def flush(self):
            sys.__stdout__.flush()

    old = sys.stdout
    sys.stdout = Tee()
    try:
        audit_features(root)
        audit_raw(root, a.min_rows)
        audit_mchat(root)
        print("\nDone. Saved to eeg_audit.txt")
    finally:
        sys.stdout = old
    Path("eeg_audit.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()