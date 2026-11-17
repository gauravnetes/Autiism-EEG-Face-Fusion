"""
prepare_eeg.py - raw EEG recordings -> window-level features for subject-wise validation.
Needs only numpy (no pandas / scipy), so it runs on your laptop and on Colab.

    python prepare_eeg.py --root .\\ASD-EEG-MCHAT-Dataset

Why it is built this way (from the audit):
  * Recording scales differ by orders of magnitude        -> only scale-independent features
  * Some channels are stuck at the 24-bit ADC limit        -> dropped per window, features averaged over good channels
  * ASD_04 and CTRL_08 have ~80-100 rows                   -> excluded
  * Recording length differs between subjects/classes      -> every subject contributes at most --max-windows windows
  * Sampling rate is unknown                               -> no frequency-band features, window length is in samples

Per window (default 512 samples, step 256): remove mean + linear trend, then per channel
  skewness, excess kurtosis, Shannon entropy (32-bin histogram), Hjorth mobility, Hjorth complexity,
  log relative variance (window variance / that recording's median window variance).
Each is averaged and std-ed over the good channels -> 12 features per window.

Outputs: eeg_windows.npz (X, y, subject, feature_names)   y: 1 = ASD, 0 = control
         eeg_subjects.csv (who was included / excluded and why)
"""
import argparse
import csv
import re
import sys
import warnings
from pathlib import Path

import numpy as np

RAIL_LO, RAIL_HI = -8388608, 8388607
EPS = 1e-12
FEATS = ["skew", "kurt", "entropy", "hjorth_mob", "hjorth_cx", "log_rel_var"]


def norm_id(name):
    u = name.upper()
    cls = "ASD" if "ASD" in u else ("CTRL" if ("CTRL" in u or "CONTROL" in u) else None)
    nums = re.findall(r"\d+", u)
    return f"{cls}_{int(nums[0]):02d}" if cls and nums else None


def channel_features(w):
    """w: (L, C) detrended window -> dict of (C,) arrays (all scale-independent except 'var')."""
    var = w.var(axis=0)
    sd = np.sqrt(var) + EPS
    z = (w - w.mean(axis=0)) / sd
    skew = (z ** 3).mean(axis=0)
    kurt = (z ** 4).mean(axis=0) - 3.0
    ent = np.empty(w.shape[1])
    for c in range(w.shape[1]):
        counts, _ = np.histogram(np.clip(z[:, c], -5, 5), bins=32, range=(-5, 5))
        p = counts[counts > 0] / counts.sum()
        ent[c] = -(p * np.log2(p)).sum()
    d1 = np.diff(w, axis=0)
    d2 = np.diff(d1, axis=0)
    v0, v1, v2 = var + EPS, d1.var(axis=0) + EPS, d2.var(axis=0) + EPS
    mob = np.sqrt(v1 / v0)
    cx = np.sqrt(v2 / v1) / mob
    return dict(skew=skew, kurt=kurt, entropy=ent, hjorth_mob=mob, hjorth_cx=cx, var=var)


def process_recording(a, win, step, min_good):
    n, c = a.shape
    t = np.arange(win, dtype=float)
    tc = t - t.mean()
    denom = tc @ tc
    rows = []                      # per valid window: dict of (C,) arrays with NaN on bad channels
    for s in range(0, n - win + 1, step):
        raw = a[s:s + win]
        bad = ((raw <= RAIL_LO) | (raw >= RAIL_HI)).any(axis=0) | (raw.std(axis=0) == 0)
        if (~bad).sum() < min_good:
            continue
        w = raw - raw.mean(axis=0)
        w = w - np.outer(tc, (tc @ w) / denom)            # remove linear trend
        f = channel_features(w)
        for k in f:
            f[k] = np.where(bad, np.nan, f[k])
        rows.append(f)
    if not rows:
        return np.zeros((0, 2 * len(FEATS)))
    V = np.array([r["var"] for r in rows])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # a fully dead channel is all-NaN; that is expected
        med = np.nanmedian(V, axis=0)
    per_ch = {k: np.array([r[k] for r in rows]) for k in FEATS[:-1]}
    per_ch["log_rel_var"] = np.log((V + EPS) / (med + EPS))
    cols = []
    for k in FEATS:
        cols.append(np.nanmean(per_ch[k], axis=1))
        cols.append(np.nanstd(per_ch[k], axis=1))
    return np.stack(cols, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=r"..\data\raw\eeg\ASD-EEG-MCHAT-Dataset")
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--step", type=int, default=256)
    ap.add_argument("--max-windows", type=int, default=60, help="max windows kept per subject (evenly spaced)")
    ap.add_argument("--min-rows", type=int, default=1000)
    ap.add_argument("--min-good", type=int, default=4, help="min good channels needed in a window")
    a = ap.parse_args()
    root = Path(a.root)
    if not root.exists():
        sys.exit(f"Path not found: {root}")

    files = [p for p in sorted(root.rglob("*.csv"))
             if p.parent.name.lower() == "eeg data" and "info" not in p.name.lower()]
    info_files = [p for p in sorted(root.rglob("*.csv"))
                  if p.parent.name.lower() == "eeg data" and "info" in p.name.lower()]
    print(f"Found {len(files)} recordings in 'EEG Data' folders.\n")

    names = [f"{FEATS[i // 2]}_{'mean' if i % 2 == 0 else 'std'}" for i in range(2 * len(FEATS))]
    X, y, subj, report, seen = [], [], [], [], set()
    for p in files:
        sid = norm_id(p.stem) or p.stem
        if sid in seen:                      # two files mapping to the same subject id: keep them apart
            sid = p.stem
        seen.add(sid)
        label = 1 if sid.startswith("ASD") else 0
        arr = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
        n = arr.shape[0]
        if n < a.min_rows:
            report.append((sid, label, n, 0, 0, f"EXCLUDED: only {n} rows"))
            continue
        F = process_recording(arr, a.win, a.step, a.min_good)
        if len(F) == 0:
            report.append((sid, label, n, 0, 0, "EXCLUDED: no valid windows"))
            continue
        idx = np.unique(np.linspace(0, len(F) - 1, min(len(F), a.max_windows)).round().astype(int))
        X.append(F[idx]); y += [label] * len(idx); subj += [sid] * len(idx)
        report.append((sid, label, n, len(F), len(idx), "ok"))

    print(f"{'subject':10s} {'class':>5s} {'rows':>7s} {'valid_win':>9s} {'used':>5s}  status")
    for sid, lab, n, nv, nu, st in report:
        print(f"{sid:10s} {'ASD' if lab else 'CTRL':>5s} {n:7d} {nv:9d} {nu:5d}  {st}")

    # cross-check against the dataset's own file list
    for ip in info_files:
        with ip.open(encoding="utf-8-sig", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            listed = [norm_id(row[0]) for row in r if row]
        missing = [s for s in listed if s and s not in seen]
        print(f"\n{ip.name} lists {len(listed)} recordings; not found as a recording file: {missing or 'none'}")

    X = np.vstack(X)
    y = np.array(y)
    subj = np.array(subj)
    if np.isnan(X).any():
        print("\nWARNING: NaNs in features; check the data.")
    ok = [r for r in report if r[5] == "ok"]
    print(f"\nUsable subjects: ASD {sum(1 for r in ok if r[1] == 1)}, control {sum(1 for r in ok if r[1] == 0)}")
    print(f"Windows: {len(y)} total (ASD {int((y == 1).sum())}, control {int((y == 0).sum())}), {X.shape[1]} features each")
    np.savez("eeg_windows.npz", X=X, y=y, subject=subj, feature_names=np.array(names))
    with open("eeg_subjects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "class", "rows", "valid_windows", "windows_used", "status"])
        for sid, lab, n, nv, nu, st in report:
            w.writerow([sid, "ASD" if lab else "CTRL", n, nv, nu, st])
    print("Saved eeg_windows.npz and eeg_subjects.csv")


if __name__ == "__main__":
    main()
