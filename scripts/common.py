"""common.py - helpers shared by the face / fusion scripts."""
import csv, os
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "valid", "test")
CLASSES = {"autistic": 1, "non_autistic": 0}      # ASD = 1, control = 0


def _norm(p):
    return str(p).replace("\\", "/").strip().lower()


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "keep")


def list_faces(root, manifest=None):
    """Return {split: (paths, labels)} sorted deterministically.
    If a manifest CSV is given, only rows with a truthy 'keep' value survive."""
    found = {}
    for sp in SPLITS:
        for cls, lab in CLASSES.items():
            d = os.path.join(root, sp, cls)
            if not os.path.isdir(d):
                raise SystemExit(f"Missing folder: {d}  (check --root; expected {{train,valid,test}}/{{autistic,non_autistic}})")
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(IMG_EXT):
                    found[(sp, cls, f.lower())] = (os.path.join(d, f), lab)

    keep = None
    if manifest:
        with open(manifest, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        cols = {c.lower(): c for c in rows[0].keys()}
        pcol = next((cols[c] for c in cols if c in ("path", "filepath", "file_path", "file", "filename", "image", "name", "relpath")), None)
        if pcol is None:
            pcol = next((cols[c] for c in cols if "path" in c or "file" in c or "name" in c), None)
        kcol = next((cols[c] for c in cols if c == "keep"), None)
        if pcol is None:
            raise SystemExit(f"Cannot find a path/filename column in manifest. Columns: {list(rows[0].keys())}")
        keep3, keepb = set(), set()
        for r in rows:
            if kcol is not None and not _truthy(r[kcol]):
                continue
            parts = _norm(r[pcol]).split("/")
            keepb.add(parts[-1])
            if len(parts) >= 3:
                keep3.add("/".join(parts[-3:]))
        use3 = any(f"{sp}/{cls}/{fn}" in keep3 for (sp, cls, fn) in found)
        keep = {k for k in found if (f"{k[0]}/{k[1]}/{k[2]}" in keep3 if use3 else k[2] in keepb)}
        print(f"[manifest] column '{pcol}', keep-column={'yes' if kcol else 'NO (all rows kept)'}, matched {len(keep)} of {len(found)} files on disk")

    out = {}
    for sp in SPLITS:
        items = sorted(k for k in found if k[0] == sp and (keep is None or k in keep))
        out[sp] = ([found[k][0] for k in items], np.array([found[k][1] for k in items], dtype=int))
    for sp in SPLITS:
        y = out[sp][1]
        print(f"  {sp:<6} ASD {int((y == 1).sum()):>5} | control {int((y == 0).sum()):>5}")
    return out


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def binary_metrics(y, prob, thr=0.5):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    pred = (np.asarray(prob) > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    auc = roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan")
    return dict(acc=(tp + tn) / len(y), bal=(sens + spec) / 2, sens=sens, spec=spec, auc=auc,
                tp=tp, tn=tn, fp=fp, fn=fn, n=len(y), k=tp + tn)
