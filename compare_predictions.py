"""
Does the pretrained ResNet-18 fix the test images that the small CNN keeps getting wrong?

    python scripts/compare_predictions.py
    python scripts/compare_predictions.py --base runs/final --new runs_resnet18/final --data strawberry_cls

For every test image it computes the error rate across all final runs of each condition
(4 optimizers x 5 seeds = 20 runs each). An image is "hard" for a condition when at least half of
its runs misclassify it. Both conditions must have been evaluated on the same test folder.

Writes  <new>/comparison/per_image_errors.csv   one row per test image, with its file name
        <new>/comparison/summary.txt            hard images per class, fixed and newly broken
"""
import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]      # repository root
IMG_EXT = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}   # same as torchvision


def per_image(pred_csv):
    """Error rate per test image, pooled over all runs, plus the same per optimizer."""
    p = pd.read_csv(pred_csv)
    p["image"] = p.groupby(["optimizer", "seed"]).cumcount()     # row order = test-set order in every run
    p["wrong"] = p.y_true != p.y_pred
    pooled = p.groupby("image").agg(true=("y_true", "first"), error_rate=("wrong", "mean"), runs=("wrong", "size"))
    by_opt = p.pivot_table(index="image", columns="optimizer", values="wrong", aggfunc="mean")
    return pooled, by_opt.add_prefix("err_")


def test_files(data):
    """File names in the order torchvision's ImageFolder reads them (sorted classes, then sorted files)."""
    root = Path(data) / "test"
    if not root.is_dir():
        return None
    return [f"{c.name}/{f.name}" for c in sorted(d for d in root.iterdir() if d.is_dir())
            for f in sorted(c.iterdir()) if f.suffix.lower() in IMG_EXT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=HERE / "runs" / "final", help="small-CNN final runs")
    ap.add_argument("--new", type=Path, default=HERE / "runs_resnet18" / "final", help="ResNet-18 final runs")
    ap.add_argument("--data", type=Path, default=HERE / "strawberry_cls", help="dataset folder (for file names)")
    ap.add_argument("--hard", type=float, default=0.5, help="error rate at which an image counts as hard")
    args = ap.parse_args()

    base, base_opt = per_image(args.base / "predictions.csv")
    new, new_opt = per_image(args.new / "predictions.csv")
    if len(base) != len(new) or not (base.true.values == new.true.values).all():
        raise SystemExit("The two conditions were not evaluated on the same test images in the same order.")
    classes = sorted(d.name for d in (args.data / "test").iterdir() if d.is_dir()) if (args.data / "test").is_dir() else None

    df = pd.DataFrame({"true": base.true, "base_error": base.error_rate, "new_error": new.error_rate})
    df = df.join(base_opt.add_prefix("base_")).join(new_opt.add_prefix("new_"))
    files = test_files(args.data)
    if files is not None and len(files) == len(df):
        df.insert(0, "file", files)
    if classes:
        df.insert(1, "class", [classes[t] for t in df.true])

    df["base_hard"] = df.base_error >= args.hard
    df["new_hard"] = df.new_error >= args.hard
    out = args.new / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_image_errors.csv", index_label="image", encoding="utf-8-sig")

    key = "class" if classes else "true"
    s = df.groupby(key).agg(test_images=("true", "size"),
                            hard_for_small_cnn=("base_hard", "sum"),
                            fixed_by_resnet=("base_hard", lambda h: int((h & ~df.loc[h.index, "new_hard"]).sum())),
                            still_hard=("base_hard", lambda h: int((h & df.loc[h.index, "new_hard"]).sum())),
                            newly_hard_for_resnet=("new_hard", lambda h: int((h & ~df.loc[h.index, "base_hard"]).sum())))
    lines = [f"'Hard' = misclassified in at least {args.hard:.0%} of a condition's runs "
             f"({int(base.runs.iloc[0])} small-CNN runs, {int(new.runs.iloc[0])} ResNet-18 runs per image).",
             "", s.to_string(), "",
             f"Mean error rate per image: small CNN {df.base_error.mean():.3f}, ResNet-18 {df.new_error.mean():.3f}"]
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()
