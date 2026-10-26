"""
=============================================================
Strawberry leaf disease — clean + crop + split  (single dataset)

Source : Strawberry Disease Detection Dataset v4 (Roboflow, YOLOv8)
         4,918 images · 280x280 letterboxed · no Roboflow augmentation
Output : strawberry_cls/  — one image + one label per crop, in
         train/validation/test  x  angular_leaf_spot / leaf_spot /
         powdery_mildew_leaf, plus manifest.csv and summary.txt

Usage:
    conda activate strawdiseaseop
    pip install pillow imagehash numpy pyyaml
    python scripts/clean_datasets.py
=============================================================
"""

import argparse
import csv
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG  — pass a dataset path on the command line to override the default
# ─────────────────────────────────────────────
HERE = Path(__file__).resolve().parents[1]
BASE_DIR = HERE / "data"

# Point to the .zip or the extracted folder. The script tries several locations.
DATASET_PATH = BASE_DIR / "Strawberry Disease Detection Dataset.v4.zip"

LETTERBOXED = True           # v4 is 280x280 "Fit (black edges)" — trim the borders
KEEP = {0: "angular_leaf_spot", 4: "leaf_spot", 6: "powdery_mildew_leaf"}
EXPECTED = {0: "angularleafspot", 4: "leafspot", 6: "powderymildewleaf"}

SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ("train", "validation", "test")
SEED = 42

CROP_MARGIN = 0.15           # extra leaf context around each box, per side
MIN_CROP_PX = 24             # skip boxes smaller than this (source pixels)
DEDUP = True                 # remove near-duplicate photos
DIHEDRAL = True              # also match flipped/rotated duplicates (slower)
HASH_THRESHOLD = 6           # max Hamming distance (of 64) = duplicate
BLACK_LEVEL = 12             # pixel value below which the letterbox edge is "black"

IMG_EXTS = {".jpg", ".jpeg", ".png"}
RF_SUFFIX = re.compile(r"_(jpg|jpeg|png)\.rf\.[0-9a-zA-Z]+$", re.IGNORECASE)


# ─────────────────────────────────────────────
# LOCATE / UNZIP THE DATASET
# ─────────────────────────────────────────────
def find_dataset(given):
    """Return a folder that directly contains data.yaml, unzipping if needed."""
    hit = None
    for p in [Path(given), BASE_DIR]:
        if p.exists():
            hit = p
            break
        # try without the .zip extension in case it's already extracted
        if p.suffix.lower() == ".zip" and p.with_suffix("").exists():
            hit = p.with_suffix("")
            break

    if hit is None:
        # search the repository root and BASE_DIR for a matching zip or folder
        for root in [HERE, BASE_DIR]:
            if not root.exists():
                continue
            for child in root.iterdir():
                if "strawberry" in child.name.lower() and "disease" in child.name.lower():
                    hit = child
                    break
            if hit:
                break

    if hit is None:
        sys.exit("❌ Could not find the dataset.\n"
                 f"   Looked for: {given}\n"
                 "   Pass the dataset ZIP or extracted folder as the first argument.")

    print(f"📁 Using: {hit}")
    if hit.is_file() and hit.suffix.lower() == ".zip":
        out = hit.with_suffix("")
        if not out.exists():
            print(f"📦 Unzipping to: {out}")
            with zipfile.ZipFile(hit) as z:
                z.extractall(out)
        else:
            print(f"📦 Already unzipped: {out}")
        hit = out

    # descend to the folder that actually holds data.yaml
    if (hit / "data.yaml").exists():
        return hit
    for y in hit.rglob("data.yaml"):
        return y.parent
    sys.exit(f"❌ No data.yaml found under {hit}. Check the extracted contents.")


def check_classes(root):
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    print(f"  classes in data.yaml: {names}")
    for cid, exp in EXPECTED.items():
        actual = names[cid] if cid < len(names) else "?"
        if re.sub(r"[^a-z]", "", str(actual).lower()) != exp:
            sys.exit(f"❌ class {cid} is '{actual}', expected '{KEEP[cid]}'.\n"
                     "   The class order in this export differs — update KEEP / "
                     "EXPECTED to match the list printed above.")
    print("  ✓ class IDs match")


# ─────────────────────────────────────────────
# LABEL PARSING + IMAGE HELPERS
# ─────────────────────────────────────────────
def group_key(stem):
    """'IMG_12_jpg.rf.ab12' -> 'IMG_12' (keeps related copies together)."""
    return RF_SUFFIX.sub("", stem)


def parse_label(path):
    """Return [(cls, cx, cy, w, h)] — handles YOLO boxes AND polygons."""
    boxes, bad = [], 0
    for line in path.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        try:
            cls = int(float(p[0]))
            v = [float(x) for x in p[1:]]
        except ValueError:
            bad += 1
            continue
        if len(v) == 4:
            cx, cy, w, h = v
        elif len(v) >= 6 and len(v) % 2 == 0:            # polygon -> its bbox
            xs, ys = v[0::2], v[1::2]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        else:
            bad += 1
            continue
        if w > 0 and h > 0:
            boxes.append((cls, cx, cy, w, h))
        else:
            bad += 1
    return boxes, bad


def content_box(img):
    """Area of real pixels (excludes letterbox black edges)."""
    if not LETTERBOXED:
        return (0, 0, img.width, img.height)
    a = np.asarray(img.convert("L")) > BLACK_LEVEL
    rows, cols = np.nonzero(a.any(axis=1))[0], np.nonzero(a.any(axis=0))[0]
    if len(rows) == 0:
        return (0, 0, img.width, img.height)
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def hashes_for(img):
    """pHash under 1 (or 8) orientations, as ints."""
    base = img.convert("RGB").resize((256, 256))
    if not DIHEDRAL:
        return [int(str(imagehash.phash(base)), 16)]
    T = Image.Transpose
    views = [base, base.transpose(T.ROTATE_90),
             base.transpose(T.ROTATE_180), base.transpose(T.ROTATE_270)]
    views += [v.transpose(T.FLIP_LEFT_RIGHT) for v in views]
    return [int(str(imagehash.phash(v)), 16) for v in views]


if hasattr(np, "bitwise_count"):
    popcount = np.bitwise_count
else:
    _LUT = np.array([bin(i).count("1") for i in range(256)], np.uint8)

    def popcount(a):
        a = np.ascontiguousarray(a)
        return _LUT[a.view(np.uint8)].reshape(a.shape + (8,)).sum(-1)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def crop_box(img, cbox, box):
    """Square-ish crop around a normalised box, clipped to real image area."""
    _, cx, cy, w, h = box
    W, H = img.size
    side = max(w * W, h * H) * (1 + 2 * CROP_MARGIN)
    x1, y1 = cx * W - side / 2, cy * H - side / 2
    x2, y2 = x1 + side, y1 + side
    x1, y1 = max(x1, cbox[0]), max(y1, cbox[1])
    x2, y2 = min(x2, cbox[2]), min(y2, cbox[3])
    if min(w * W, h * H) < MIN_CROP_PX or x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return img.crop((round(x1), round(y1), round(x2), round(y2)))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Clean, deduplicate, and split strawberry leaf crops.")
    parser.add_argument("dataset", nargs="?", type=Path, default=DATASET_PATH,
                        help="YOLO dataset ZIP or extracted folder (default: data/)")
    parser.add_argument("--out", type=Path, default=HERE / "strawberry_cls",
                        help="output crop directory (default: repository-root strawberry_cls/)")
    args = parser.parse_args()
    print("=" * 60 + "\n🍓 Clean → 3 leaf-disease classes  (single dataset)\n" + "=" * 60)
    root = find_dataset(args.dataset)
    check_classes(root)

    out_dir = args.out
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"❌ {out_dir} already exists. Delete or rename it, then re-run.")

    # 1–2. scan + filter -------------------------------------------------
    drops = Counter()
    groups = defaultdict(list)
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
        if not img_dir.exists():
            continue
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        for ip in imgs:
            drops["scanned"] += 1
            lp = lbl_dir / (ip.stem + ".txt")
            if not lp.exists():
                drops["no_label_file"] += 1
                continue
            boxes, bad = parse_label(lp)
            drops["bad_label_lines"] += bad
            kept = [b for b in boxes if b[0] in KEEP]
            if not kept:
                drops["no_selected_class"] += 1
                continue
            try:
                with Image.open(ip) as im:
                    im.verify()
            except Exception:
                drops["corrupt_image"] += 1
                continue
            groups[group_key(ip.stem)].append(
                {"orig_split": split_dir.name, "img": ip, "boxes": kept})
        print(f"  {split_dir.name}: {len(imgs)} images scanned")
    print(f"  → {len(groups)} photos keep at least one selected class")

    # 3. collapse any name-sharing copies to one image -------------------
    for g in groups:
        groups[g] = groups[g][:1]

    # 4. near-duplicate removal ------------------------------------------
    gids = sorted(groups)
    kept_groups = gids
    dup_pairs = []
    if DEDUP and len(gids) > 1:
        print(f"🔍 Hashing {len(gids)} photos for duplicates"
              f"{' (8 orientations)' if DIHEDRAL else ''}...")
        H = []
        for g in gids:
            with Image.open(groups[g][0]["img"]) as im:
                im = im.convert("RGB")
                im = im.crop(content_box(im))
                H.append(hashes_for(im))
        H = np.array(H, dtype=np.uint64)

        uf = UnionFind(len(gids))
        for i in range(len(gids) - 1):
            d = popcount(H[i + 1:] ^ H[i, 0]).min(axis=1)
            for j in np.nonzero(d <= HASH_THRESHOLD)[0]:
                j = i + 1 + int(j)
                uf.union(i, j)
                dup_pairs.append((gids[i], gids[j], int(d[j - i - 1])))

        clusters = defaultdict(list)
        for i in range(len(gids)):
            clusters[uf.find(i)].append(i)
        kept_groups = sorted(gids[min(m)] for m in clusters.values())
        drops["duplicate_photos_removed"] = len(gids) - len(kept_groups)
        print(f"  {len(dup_pairs)} duplicate pairs → {len(kept_groups)} unique photos")

    # 5. grouped, class-balanced split -----------------------------------
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for g in kept_groups:
        c = Counter(b[0] for b in groups[g][0]["boxes"])
        by_class[c.most_common(1)[0][0]].append(g)
    split_of = {}
    for cls in sorted(by_class):
        units = by_class[cls]
        rng.shuffle(units)
        n = len(units)
        n_tr, n_va = round(n * SPLIT_RATIOS[0]), round(n * SPLIT_RATIOS[1])
        for k, g in enumerate(units):
            split_of[g] = SPLIT_NAMES[0 if k < n_tr else 1 if k < n_tr + n_va else 2]

    # 6. write crops + manifest ------------------------------------------
    for s in SPLIT_NAMES:
        for name in KEEP.values():
            (out_dir / s / name).mkdir(parents=True, exist_ok=True)
    new_id = {cid: i for i, cid in enumerate(KEEP)}
    crop_counts, photo_counts, rows = Counter(), Counter(), []

    for g in kept_groups:
        split = split_of[g]
        photo_counts[split] += 1
        rec = groups[g][0]
        with Image.open(rec["img"]) as im:
            im = im.convert("RGB")
            cbox = content_box(im)
            for k, box in enumerate(rec["boxes"]):
                crop = crop_box(im, cbox, box)
                if crop is None:
                    drops["box_too_small"] += 1
                    continue
                label = KEEP[box[0]]
                fname = f"{rec['img'].stem}_c{k}.jpg"
                crop.save(out_dir / split / label / fname, quality=95)
                crop_counts[(split, label)] += 1
                rows.append({
                    "crop_path": f"{split}/{label}/{fname}",
                    "split": split, "label": label, "label_id": new_id[box[0]],
                    "orig_class": box[0], "orig_split": rec["orig_split"],
                    "source_image": rec["img"].name, "group_id": g, "box_index": k,
                    "cx": box[1], "cy": box[2], "w": box[3], "h": box[4],
                    "crop_w": crop.width, "crop_h": crop.height,
                })

    if not rows:
        sys.exit("❌ No crops were written — check the class IDs and boxes.")
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kept", "duplicate", "hamming_distance"])
        w.writerows(dup_pairs)

    # summary ------------------------------------------------------------
    lines = ["DROPS / COUNTS"] + [f"  {k:26s} {v}" for k, v in drops.items()]
    header = f"{'split':12s}" + "".join(f"{n:>22s}" for n in KEEP.values()) + f"{'photos':>10s}"
    lines += ["", header]
    for s in SPLIT_NAMES:
        lines.append(f"{s:12s}"
                     + "".join(f"{crop_counts[(s, n)]:>22d}" for n in KEEP.values())
                     + f"{photo_counts[s]:>10d}")
    tot = f"{'TOTAL':12s}" + "".join(
        f"{sum(crop_counts[(s, n)] for s in SPLIT_NAMES):>22d}" for n in KEEP.values())
    lines += [tot, "", f"Total crops: {len(rows)}   Unique photos: {len(kept_groups)}"]
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary)
    print("\n" + "=" * 60 + "\n" + summary + "\n" + "=" * 60)
    print(f"✅ Output: {out_dir}")


if __name__ == "__main__":
    main()
