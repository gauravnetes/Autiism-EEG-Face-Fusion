"""
prepare_eeg_v2.py - raw EEG -> window features, with extra feature groups.

Groups (each scale-/offset-invariant):
  base : skew, kurtosis, Shannon entropy, Hjorth mobility/complexity, log relative variance   (12 feats)
  dwt  : db4 wavelet, 5 levels: relative energy per level + wavelet entropy (mean/std over ch) (14 feats)
  conn : inter-channel correlation: mean|r|, std|r|, mean r, top-eigenvalue share            ( 4 feats)
  spec : relative band power (only if --fs is given; sampling rate is unknown in this dataset) (10 feats)

Usage:
  pip install PyWavelets scipy scikit-learn pandas
  python prepare_eeg_v2.py --root ./ASD-EEG-MCHAT-Dataset [--fs 250] [--channels 1 2 3 4 5 6 7 8]
"""
import argparse, glob, os, re, warnings
import numpy as np
import pandas as pd
import pywt
from scipy.signal import detrend, welch
from scipy.stats import skew, kurtosis

DEAD = -8388608          # 24-bit ADC floor = unconnected / saturated electrode
W, S = 512, 256          # window length, step (same as v1)
EPS = 1e-12
BANDS = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30), ("gamma", 30, 45)]


def read_recording(path, channels):
    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number])
    if channels:                                   # explicit 0-based column positions
        cols = [num.columns[i] for i in channels]
    else:
        junk = re.compile(r"time|index|sample|stamp|label|class|trigger|marker", re.I)
        cols = [c for c in num.columns if not junk.search(str(c))]
        if len(cols) > 8:
            cols = cols[-8:]
    return num[cols].to_numpy(float), [str(c) for c in cols]


def ms(v):                                         # mean and std across channels
    return [float(np.mean(v)), float(np.std(v))]


def feats_base(x, med_g):
    n = len(x)
    out = []
    sk, ku = skew(x, axis=0), kurtosis(x, axis=0)
    z = (x - x.mean(0)) / (x.std(0) + EPS)
    ent = []
    for c in range(x.shape[1]):
        h, _ = np.histogram(z[:, c], bins=32)
        p = h[h > 0] / n
        ent.append(-(p * np.log(p)).sum())
    d1, d2 = np.diff(x, axis=0), np.diff(x, 2, axis=0)
    v0, v1, v2 = x.var(0), d1.var(0) + EPS, d2.var(0) + EPS
    mob = np.sqrt(v1 / (v0 + EPS))
    comp = np.sqrt(v2 / v1) / (mob + EPS)
    lrv = np.log((v0 + EPS) / (med_g + EPS))
    for v in (sk, ku, np.array(ent), mob, comp, lrv):
        out += ms(v)
    return out


def feats_dwt(x):
    co = pywt.wavedec(x, "db4", level=5, axis=0)
    e = np.stack([(c ** 2).sum(0) for c in co])    # (6, nch)
    rel = e / (e.sum(0) + EPS)
    went = -(rel * np.log(rel + EPS)).sum(0)
    out = []
    for k in range(rel.shape[0]):
        out += ms(rel[k])
    return out + ms(went)


def feats_conn(x):
    r = np.corrcoef(x.T)
    iu = np.triu_indices(r.shape[0], k=1)
    v = r[iu]
    top = np.linalg.eigvalsh(r)[-1] / r.shape[0]
    return [float(np.abs(v).mean()), float(np.abs(v).std()), float(v.mean()), float(top)]


def feats_spec(x, fs):
    f, p = welch(x, fs=fs, nperseg=min(256, len(x)), axis=0)
    tot = p[(f >= 1) & (f < min(45, fs / 2))].sum(0) + EPS
    out = []
    for _, lo, hi in BANDS:
        m = (f >= lo) & (f < min(hi, fs / 2))
        out += ms(p[m].sum(0) / tot) if m.any() else [0.0, 0.0]
    return out


NAMES = {
    "base": [f"{n}_{s}" for n in ["skew", "kurt", "entropy", "mobility", "complexity", "logrelvar"] for s in ("mean", "std")],
    "dwt": [f"dwt_rel{k}_{s}" for k in range(6) for s in ("mean", "std")] + ["dwt_entropy_mean", "dwt_entropy_std"],
    "conn": ["corr_absmean", "corr_absstd", "corr_mean", "corr_topeig"],
    "spec": [f"band_{b}_{s}" for b, _, _ in BANDS for s in ("mean", "std")],
}


def process(path, a):
    raw, cols = read_recording(path, a.channels)
    n = len(raw)
    if n < a.min_rows:
        return None, n, cols
    starts = np.arange(0, n - W + 1, S)
    segs = np.stack([raw[s:s + W] for s in starts])               # (nw, W, nch)
    bad = (segs == DEAD).any(1) | np.isnan(segs).any(1) | (np.ptp(segs, axis=1) == 0)
    ok = (~bad).sum(1) >= a.min_good
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        det = detrend(np.where(np.isnan(segs), 0.0, segs), axis=1)  # mean + linear trend removed
        var = det.var(1)
        var[bad] = np.nan
        med = np.nanmedian(var[ok], axis=0)
    idx = np.flatnonzero(ok)
    if len(idx) > a.cap:
        idx = idx[np.unique(np.round(np.linspace(0, len(idx) - 1, a.cap)).astype(int))]
    rows = []
    for i in idx:
        g = ~bad[i]
        x = det[i][:, g]
        f = feats_base(x, med[g]) + feats_dwt(x) + feats_conn(x)
        if a.fs:
            f += feats_spec(x, a.fs)
        if np.all(np.isfinite(f)):
            rows.append(f)
    return np.array(rows), n, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fs", type=float, default=None, help="sampling rate in Hz, enables spectral features")
    ap.add_argument("--channels", type=int, nargs="+", default=None, help="0-based numeric column positions of the 8 EEG channels")
    ap.add_argument("--min_rows", type=int, default=2000)
    ap.add_argument("--min_good", type=int, default=4)
    ap.add_argument("--cap", type=int, default=60, help="max windows per child (stops long recordings dominating)")
    ap.add_argument("--out", default="eeg_windows_v2.npz")
    a = ap.parse_args()

    names = NAMES["base"] + NAMES["dwt"] + NAMES["conn"] + (NAMES["spec"] if a.fs else [])
    groups = (["base"] * 12 + ["dwt"] * 14 + ["conn"] * 4 + (["spec"] * 10 if a.fs else []))
    X, child, lab, report = [], [], {}, []
    for tag, folder, y in (("ASD", "Raw ASD Data", 1), ("CTRL", "Raw Control Data", 0)):
        files = [q for q in glob.glob(os.path.join(a.root, folder, "EEG Data", "*.csv"))
                 if "info" not in os.path.basename(q).lower()]          # any recording name; skip *_File_Info.csv
        print(f"[{tag}] {len(files)} recording files:", sorted(os.path.basename(q) for q in files))
        for p in sorted(files, key=lambda q: int(re.search(r"(\d+)", os.path.basename(q)).group(1))):
            cid = f"{tag}_{int(re.search(r'(\d+)', os.path.basename(p)).group(1)):02d}"
            f, n, cols = process(p, a)
            if f is None or len(f) == 0:
                report.append(f"EXCLUDED {cid}: {n} rows"); continue
            X.append(f); child += [cid] * len(f); lab[cid] = y
            report.append(f"{cid}: rows={n} windows={len(f)} columns={cols}")
    print("\n".join(report))
    if not X:
        raise SystemExit("No recordings found - check --root and folder names.")
    X = np.vstack(X)
    ids = sorted(lab)
    np.savez(a.out, X=X, child=np.array(child), ids=np.array(ids),
             labels=np.array([lab[i] for i in ids]), names=np.array(names), groups=np.array(groups))
    n1 = sum(lab.values())
    print(f"\n{len(ids)} children (ASD {n1}, control {len(ids) - n1}), {len(X)} windows, {X.shape[1]} features -> {a.out}")
    print("CHECK the printed 'columns=' list: it must be exactly the 8 EEG channels (no time/index column).")


if __name__ == "__main__":
    main()
