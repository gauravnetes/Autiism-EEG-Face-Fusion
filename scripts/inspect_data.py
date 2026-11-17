"""
inspect_data.py  -  sanity checks for the ASD face + EEG project (Windows friendly)

USAGE (PowerShell, from the results directory):
    pip install pillow numpy
    # Optional: openpyxl for .xlsx files, pandas for legacy .xls files
    python ..\\scripts\\inspect_data.py faces --root ..\\data\\raw\\faces\\Autistic_Children_Facial_Image
    python ..\\scripts\\inspect_data.py eeg   --root ..\\data\\raw\\eeg\\ASD-EEG-MCHAT-Dataset

faces : counts per split/class, unreadable files, exact + near-duplicate detection
        (within split, ACROSS splits = leakage, and across classes = label conflicts),
        simple shortcut check (image size / brightness / file size per class),
        and writes:
            face_duplicates.csv        every flagged pair
            faces_clean_manifest.csv   every image with keep=True/False
                                        (cross-split copies are dropped from the later split,
                                         within-split copies keep one)
            faces_report.txt           the printed summary
eeg   : walks the Mendeley folder, lists every file, and for csv/xlsx files prints shape,
        columns, dtypes and the first rows so we can see the real structure.
        Writes eeg_inventory.txt  (send me this file or a screenshot).
"""
import argparse
import hashlib
import io
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train": "train", "training": "train",
               "valid": "valid", "val": "valid", "validation": "valid",
               "test": "test"}
SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2, "unknown": 3}


# --------------------------------------------------------------------------- faces
def dhash_bits(img, size=8):
    """64-bit difference hash as a 0/1 numpy vector (robust to resizing / recompression)."""
    import numpy as np

    g = img.convert("L").resize((size + 1, size))
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.uint8)


def find_images(root: Path):
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts
        split = next((SPLIT_NAMES[x.lower()] for x in rel_parts[:-1] if x.lower() in SPLIT_NAMES), "unknown")
        label = p.parent.name
        rows.append((p, split, label))
    return rows


def run_faces(root: Path, near_thresh: int):
    import numpy as np
    from PIL import Image

    items = find_images(root)
    if not items:
        sys.exit(f"No images found under {root}. Check the --root path.")

    print(f"Root: {root}")
    print(f"Total image files: {len(items)}\n")

    # --- counts
    counts = Counter((s, l) for _, s, l in items)
    print("== Files per split / class ==")
    for (s, l), n in sorted(counts.items(), key=lambda kv: (SPLIT_ORDER.get(kv[0][0], 9), kv[0][1])):
        print(f"  {s:8s} {l:15s} {n}")
    print()

    # --- read every image once: md5, dhash, size, brightness
    recs, bad = [], []
    for p, split, label in items:
        try:
            data = p.read_bytes()
            md5 = hashlib.md5(data).hexdigest()
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                w, h = im.size
                bits = dhash_bits(im)
                bright = float(np.asarray(im.convert("L"), dtype=np.float32).mean())
            recs.append(dict(path=p, split=split, label=label, md5=md5, bits=bits,
                             w=w, h=h, bright=bright, kb=len(data) / 1024))
        except Exception as e:  # corrupt / unreadable
            bad.append((p, str(e)))
    print(f"Unreadable/corrupt files: {len(bad)}")
    for p, e in bad[:10]:
        print(f"  {p}  ->  {e}")
    print()

    n = len(recs)
    key = lambda r: (SPLIT_ORDER.get(r["split"], 9), str(r["path"]))

    # --- exact duplicates (md5)
    by_md5 = defaultdict(list)
    for i, r in enumerate(recs):
        by_md5[r["md5"]].append(i)
    exact_groups = [g for g in by_md5.values() if len(g) > 1]

    # --- near duplicates (dHash hamming distance)
    H = np.stack([r["bits"] for r in recs]).astype(np.float32)          # (n, 64)
    dist = H @ (1 - H).T + (1 - H) @ H.T                                  # hamming matrix
    ii, jj = np.where(np.triu(dist <= near_thresh, k=1))

    pairs = {}
    for i, j in zip(ii, jj):
        pairs[(i, j)] = ("near", int(dist[i, j]))
    for g in exact_groups:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                i, j = sorted((g[a], g[b]))
                pairs[(i, j)] = ("exact", 0)

    def kind(i, j):
        a, b = recs[i], recs[j]
        if a["label"] != b["label"]:
            return "LABEL_CONFLICT (same image, different class)"
        if a["split"] != b["split"]:
            return "CROSS_SPLIT_LEAK"
        return "within_split"

    tally = Counter(kind(i, j) for (i, j) in pairs)
    print(f"== Duplicate check (exact md5 + near dHash distance <= {near_thresh}) ==")
    print(f"  flagged pairs: {len(pairs)}")
    for k, v in tally.most_common():
        print(f"  {k}: {v}")
    print()

    # --- write pairs csv
    import csv
    with open("face_duplicates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "hamming", "category", "split_a", "label_a", "file_a", "split_b", "label_b", "file_b"])
        for (i, j), (t, d) in sorted(pairs.items()):
            a, b = recs[i], recs[j]
            w.writerow([t, d, kind(i, j), a["split"], a["label"], a["path"], b["split"], b["label"], b["path"]])

    # --- clean manifest: keep earliest split / first file of every duplicate cluster
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j) in pairs:
        parent[find(i)] = find(j)
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    keep = [True] * n
    conflict_clusters = 0
    for members in clusters.values():
        if len(members) == 1:
            continue
        if len({recs[m]["label"] for m in members}) > 1:
            conflict_clusters += 1
            for m in members:          # same face labelled both ways -> unusable, drop all
                keep[m] = False
            continue
        members.sort(key=lambda m: key(recs[m]))
        for m in members[1:]:
            keep[m] = False

    with open("faces_clean_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["keep", "split", "label", "path"])
        for r, k in zip(recs, keep):
            w.writerow([k, r["split"], r["label"], r["path"]])

    kept = Counter((r["split"], r["label"]) for r, k in zip(recs, keep) if k)
    print("== After removing duplicates (faces_clean_manifest.csv) ==")
    print(f"  kept {sum(keep)} of {n} images; clusters with conflicting labels dropped: {conflict_clusters}")
    for (s, l), c in sorted(kept.items(), key=lambda kv: (SPLIT_ORDER.get(kv[0][0], 9), kv[0][1])):
        print(f"  {s:8s} {l:15s} {c}")
    print()

    # --- shortcut check
    print("== Shortcut check: do the classes differ in trivial properties? ==")
    print("   (big gaps in size / brightness / file size = the model may learn the photo style, not the face)")
    by_label = defaultdict(list)
    for r in recs:
        by_label[r["label"]].append(r)
    print(f"  {'class':15s} {'n':>5s} {'median WxH':>14s} {'brightness':>11s} {'file KB':>9s}")
    for l, rs in sorted(by_label.items()):
        mw, mh = np.median([r["w"] for r in rs]), np.median([r["h"] for r in rs])
        print(f"  {l:15s} {len(rs):5d} {int(mw):>6d}x{int(mh):<6d} "
              f"{np.mean([r['bright'] for r in rs]):11.1f} {np.median([r['kb'] for r in rs]):9.1f}")
    sizes = Counter((r["label"], f"{r['w']}x{r['h']}") for r in recs)
    print("  most common image sizes per class:")
    for l in sorted(by_label):
        top = [(s, c) for (lab, s), c in sizes.most_common() if lab == l][:3]
        print(f"    {l}: {top}")
    print("\nDone. Files written: face_duplicates.csv, faces_clean_manifest.csv, faces_report.txt")


# --------------------------------------------------------------------------- eeg
def run_eeg(root: Path, max_files_detail: int, rows: int):
    import csv

    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    print(f"Root: {root}")
    print(f"Total files: {len(files)}\n")

    ext_count = Counter(p.suffix.lower() for p in files)
    print("== File types ==")
    for e, c in ext_count.most_common():
        print(f"  {e or '(none)':8s} {c}")
    print()

    print("== Folder summary ==")
    by_dir = defaultdict(list)
    for p in files:
        by_dir[p.parent.relative_to(root)].append(p)
    for d, ps in sorted(by_dir.items(), key=lambda kv: str(kv[0])):
        total_mb = sum(p.stat().st_size for p in ps) / 1e6
        print(f"  {str(d) or '.'}: {len(ps)} files, {total_mb:.1f} MB, e.g. {[p.name for p in ps[:4]]}")
    print()

    detailed = 0
    print("== Tabular files (csv / xlsx / xls) ==")
    for p in files:
        if p.suffix.lower() not in {".csv", ".xlsx", ".xls", ".tsv"}:
            continue
        if detailed >= max_files_detail:
            print(f"  ... (more tabular files not shown; raise --max-files)")
            break
        detailed += 1
        rel = p.relative_to(root)
        print(f"\n--- {rel}  ({p.stat().st_size / 1024:.0f} KB)")
        try:
            if p.suffix.lower() in {".csv", ".tsv"}:
                with p.open("r", encoding="utf-8-sig", newline="") as f:
                    sample = f.read(8192)
                    f.seek(0)
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
                    except csv.Error:
                        dialect = csv.excel_tab if p.suffix.lower() == ".tsv" else csv.excel
                    reader = csv.reader(f, dialect)
                    header = next(reader, [])
                    data = []
                    for row in reader:
                        data.append(row)
                        if len(data) >= 5000:
                            break
                sheets = {"": (header, data)}
            elif p.suffix.lower() == ".xlsx":
                from openpyxl import load_workbook

                workbook = load_workbook(p, read_only=True, data_only=True)
                sheets = {}
                for worksheet in workbook.worksheets:
                    values = worksheet.iter_rows(values_only=True)
                    header = list(next(values, ()))
                    data = [list(row) for _, row in zip(range(5000), values)]
                    sheets[worksheet.title] = (header, data)
                workbook.close()
            else:
                import pandas as pd

                sheets = {
                    name: (list(df.columns), df.values.tolist())
                    for name, df in pd.read_excel(p, sheet_name=None, nrows=5000).items()
                }

            for name, (header, data) in sheets.items():
                shape = (len(data), len(header))
                print(f"  sheet '{name}'  shape (first 5000 rows read): {shape}" if name else f"  shape (first 5000 rows read): {shape}")
                print(f"  columns: {header[:30]}")
                dtype_counts = Counter()
                for col_index in range(len(header)):
                    values = [row[col_index] if col_index < len(row) else "" for row in data]
                    present = [str(value).strip() for value in values if value is not None and str(value).strip()]
                    if not present:
                        dtype = "object"
                    else:
                        try:
                            for value in present:
                                int(value)
                            dtype = "int64" if len(present) == len(values) else "float64"
                        except ValueError:
                            try:
                                for value in present:
                                    float(value)
                                dtype = "float64"
                            except ValueError:
                                dtype = "object"
                    dtype_counts[dtype] += 1
                print("  dtypes:", dict(dtype_counts))
                print("  preview:")
                print(f"    {header[:30]}")
                for row in data[:rows]:
                    print(f"    {row[:30]}")
                low = [i for i, c in enumerate(header)
                       if str(c).lower() in {"label", "class", "group", "target", "diagnosis", "asd", "y"}]
                for i in low:
                    values = [str(row[i]) for row in data if i < len(row) and row[i] is not None and str(row[i]).strip()]
                    print(f"  value counts for '{header[i]}': {dict(Counter(values).most_common())}")
        except Exception as e:
            print(f"  could not read: {e}")

    print("\n== Non-tabular files (first 25) ==")
    other = [p for p in files if p.suffix.lower() not in {".csv", ".xlsx", ".xls", ".tsv"}]
    for p in other[:25]:
        print(f"  {p.relative_to(root)}  ({p.stat().st_size / 1024:.0f} KB)")
    if len(other) > 25:
        print(f"  ... and {len(other) - 25} more")
    print("\nDone. Report saved to eeg_inventory.txt")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("faces")
    f.add_argument("--root", default=r"..\data\raw\faces\Autistic_Children_Facial_Image")
    f.add_argument("--near", type=int, default=4, help="max dHash Hamming distance counted as near-duplicate (0-64)")

    e = sub.add_parser("eeg")
    e.add_argument("--root", default=r"..\data\raw\eeg\ASD-EEG-MCHAT-Dataset")
    e.add_argument("--max-files", type=int, default=12, help="how many tabular files to show in detail")
    e.add_argument("--rows", type=int, default=5, help="preview rows per table")

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

    with redirect_stdout(Tee()):
        if a.cmd == "faces":
            run_faces(root, a.near)
        else:
            run_eeg(root, a.max_files, a.rows)

    out = "faces_report.txt" if a.cmd == "faces" else "eeg_inventory.txt"
    Path(out).write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
