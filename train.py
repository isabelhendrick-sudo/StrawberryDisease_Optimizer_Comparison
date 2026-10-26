"""
=============================================================
Strawberry leaf disease — optimizer comparison (train.py)

One fixed small CNN, trained from the same starting weights with
SGD, Adagrad, RMSprop and Adam. Everything except the optimizer is
held constant: data split, preprocessing, augmentation, batch order,
loss, epoch budget and checkpoint rule.

Input : strawberry_cls/  (made by scripts/clean_datasets.py, uses manifest.csv)
Output: runs/<timestamp>_<tag>/
          results.csv                     one row per run (the summary table)
          compare_seed<N>.png             all optimizers' val curves overlaid
          <optimizer>_seed<N>/
              history.csv                 per-epoch loss / acc / macro-F1 / time
              curves.png                  learning curves
              confusion_matrix.png        test-set confusion matrix
              test_report.json            precision / recall / F1, macro + weighted
              test_predictions.csv        every test crop with its prediction
              config.json                 exact settings used
              best.pt                     best checkpoint (by val macro-F1)

Usage:
    conda activate AI_strawberry_env
    python scripts/pilot/train.py                                   # pilot: 4 optimizers, seed 0, 5 epochs
    python scripts/pilot/train.py --optimizer adam --epochs 5       # a single optimizer
    python scripts/pilot/train.py --seeds 0 1 2 --epochs 40 --tag final   # full experiment
=============================================================
"""

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HERE = Path(__file__).resolve().parents[2]
DEFAULT_DATA = HERE / "strawberry_cls"
CLASSES = ["angular_leaf_spot", "leaf_spot", "powdery_mildew_leaf"]   # label_id 0, 1, 2
SPLITS = ("train", "validation", "test")

# Standard PyTorch defaults (plain SGD = no momentum). State these in your report.
DEFAULT_LR = {"sgd": 0.01, "adagrad": 0.01, "rmsprop": 0.001, "adam": 0.001}
OPTIMIZERS = list(DEFAULT_LR)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def find_data(given):
    cands = ([Path(given)] if given else []) + [
        Path(DEFAULT_DATA),
        HERE / "strawberry_cls",
        Path.cwd() / "strawberry_cls",
    ]
    for c in cands:
        if (c / "manifest.csv").exists():
            return c
    sys.exit("❌ Could not find strawberry_cls/manifest.csv.\n"
             "   Run with:  python scripts/pilot/train.py --data \"C:/path/to/strawberry_cls\"")


def load_arrays(root, size):
    """Load every crop once, resized to size x size, and cache as .npz."""
    manifest = root / "manifest.csv"
    cache = root / f"cache_{size}.npz"
    if cache.exists() and cache.stat().st_mtime > manifest.stat().st_mtime:
        d = np.load(cache)
        print(f"📦 Loaded cached images: {cache.name}")
        return {s: (d[f"{s}_x"], d[f"{s}_y"]) for s in SPLITS}

    print(f"📦 Reading crops and resizing to {size}x{size} (cached for next time)...")
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out, missing = {}, 0
    for s in SPLITS:
        xs, ys = [], []
        for r in (r for r in rows if r["split"] == s):
            p = root / r["crop_path"]
            if not p.exists():                     # crop deleted during manual review
                missing += 1
                continue
            with Image.open(p) as im:
                xs.append(np.asarray(im.convert("RGB").resize((size, size), Image.BICUBIC)))
            ys.append(int(r["label_id"]))
        out[s] = (np.stack(xs).astype(np.uint8), np.array(ys, dtype=np.int64))
    if missing:
        print(f"  ⚠️  {missing} manifest rows had no image file (skipped)")
    np.savez(cache, **{f"{s}_x": out[s][0] for s in SPLITS},
             **{f"{s}_y": out[s][1] for s in SPLITS})
    return out


def augment(x):
    """Random flip + 90° rotation per image (all 8 orientations). Training only."""
    b = x.size(0)
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    k = torch.randint(0, 4, (b,), device=x.device)
    out = x.clone()
    for r in (1, 2, 3):
        m = k == r
        if m.any():
            out[m] = torch.rot90(x[m], r, dims=(2, 3))
    return out


# ─────────────────────────────────────────────
# MODEL  (fixed across all runs)
# ─────────────────────────────────────────────
class SmallCNN(nn.Module):
    """4 conv blocks (16-32-64-128, 3x3, ReLU, max-pool) → GAP → FC64 → dropout → 3."""

    def __init__(self, n_classes=3, dropout=0.3):
        super().__init__()
        chans, layers = [3, 16, 32, 64, 128], []
        for cin, cout in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def make_optimizer(name, params, lr):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(name)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def predict(model, X, prep, bs):
    model.eval()
    logits = [model(prep(X[i:i + bs])) for i in range(0, len(X), bs)]
    return torch.cat(logits)


def evaluate(model, X, y, prep, crit, bs):
    logits = predict(model, X, prep, bs)
    loss = crit(logits, y).item()
    pred = logits.argmax(1).cpu().numpy()
    true = y.cpu().numpy()
    return {"loss": loss, "acc": accuracy_score(true, pred),
            "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
            "pred": pred, "true": true, "logits": logits}


def warm_up(device, size):
    """Run a throwaway model once so CUDA start-up cost isn't charged to the first optimizer."""
    if device.type != "cuda":
        return
    m = SmallCNN().to(device)
    o = torch.optim.SGD(m.parameters(), lr=0.01)
    x = torch.randn(32, 3, size, size, device=device)
    for _ in range(10):
        o.zero_grad()
        m(x).sum().backward()
        o.step()
    sync(device)


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def plot_curves(hist, title, path):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], "o-", label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], "o-", label="validation")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[1].plot(ep, [h["train_acc"] for h in hist], "o-", label="train accuracy")
    ax[1].plot(ep, [h["val_acc"] for h in hist], "o-", label="val accuracy")
    ax[1].plot(ep, [h["val_macro_f1"] for h in hist], "s--", label="val macro-F1")
    ax[1].set(title="Accuracy / macro-F1", xlabel="epoch", ylim=(0, 1))
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    short = ["angular\nleaf spot", "leaf\nspot", "powdery\nmildew"]
    ax.set_xticks(range(3), short)
    ax.set_yticks(range(3), short)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_compare(histories, seed, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, hist in histories.items():
        ep = [h["epoch"] for h in hist]
        ax[0].plot(ep, [h["val_loss"] for h in hist], "o-", label=name)
        ax[1].plot(ep, [h["val_macro_f1"] for h in hist], "o-", label=name)
    ax[0].set(title="Validation loss", xlabel="epoch")
    ax[1].set(title="Validation macro-F1", xlabel="epoch", ylim=(0, 1))
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.suptitle(f"Optimizer comparison (seed {seed})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────
# ONE RUN = one optimizer, one seed
# ─────────────────────────────────────────────
def run_one(opt_name, seed, T, prep, class_w, args, device, session):
    run_dir = session / f"{opt_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    lr = args.lr if args.lr is not None else DEFAULT_LR[opt_name]

    set_seed(seed)                               # same seed → identical starting weights
    model = SmallCNN(dropout=args.dropout).to(device)
    opt = make_optimizer(opt_name, model.parameters(), lr)
    crit = nn.CrossEntropyLoss(weight=class_w)
    order = torch.Generator().manual_seed(seed)  # same batch order for every optimizer

    Xtr, ytr = T["train"]
    Xva, yva = T["validation"]
    n, bs = len(Xtr), args.batch_size
    json.dump({"optimizer": opt_name, "lr": lr, "seed": seed, "epochs": args.epochs,
               "batch_size": bs, "img_size": args.img_size, "dropout": args.dropout,
               "patience": args.patience,
               "class_weights": class_w.tolist() if class_w is not None else None,
               "device": str(device), "n_train": n, "n_val": len(Xva),
               "n_test": len(T["test"][0])},
              open(run_dir / "config.json", "w"), indent=2)

    print(f"\n▶ {opt_name.upper():8s} seed {seed}  lr {lr}")
    hist, best_f1, best_ep, best_state, bad = [], -1.0, 0, None, 0
    sync(device)
    t_run = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        model.train()
        sync(device)
        t0 = time.perf_counter()
        perm = torch.randperm(n, generator=order).to(device)
        loss_sum, correct = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = augment(prep(Xtr[idx])), ytr[idx]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
        sync(device)
        train_time = time.perf_counter() - t0

        v = evaluate(model, Xva, yva, prep, crit, bs)
        sync(device)
        epoch_time = time.perf_counter() - t0
        hist.append({"epoch": ep, "train_loss": loss_sum / n, "train_acc": correct / n,
                     "val_loss": v["loss"], "val_acc": v["acc"], "val_macro_f1": v["macro_f1"],
                     "train_time_s": train_time, "epoch_time_s": epoch_time})
        flag = ""
        if v["macro_f1"] > best_f1:
            best_f1, best_ep, bad = v["macro_f1"], ep, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            flag = "  ★ best"
        else:
            bad += 1
        print(f"  ep {ep:3d}  train loss {loss_sum / n:.3f} acc {correct / n:.3f}  |  "
              f"val loss {v['loss']:.3f} acc {v['acc']:.3f} F1 {v['macro_f1']:.3f}  |  "
              f"{epoch_time:5.1f}s{flag}")
        if bad >= args.patience:
            print(f"  ⏹ early stop: no val macro-F1 gain for {args.patience} epochs")
            break

    sync(device)
    total_time = time.perf_counter() - t_run

    # ---- final evaluation with the best checkpoint --------------------
    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best.pt")
    res = {s: evaluate(model, *T[s], prep, crit, bs) for s in SPLITS}
    te = res["test"]
    report = classification_report(te["true"], te["pred"], labels=[0, 1, 2],
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(te["true"], te["pred"], labels=[0, 1, 2])
    json.dump({"classification_report": report, "confusion_matrix": cm.tolist()},
              open(run_dir / "test_report.json", "w"), indent=2)

    probs = torch.softmax(te["logits"], 1).cpu().numpy()
    with open(run_dir / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "true", "pred"] + [f"p_{c}" for c in CLASSES])
        for i, (t, p) in enumerate(zip(te["true"], te["pred"])):
            w.writerow([i, CLASSES[t], CLASSES[p]] + [f"{x:.4f}" for x in probs[i]])

    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)

    label = f"{opt_name.upper()} (seed {seed})"
    plot_curves(hist, label, run_dir / "curves.png")
    plot_confusion(cm, f"{label} — test set", run_dir / "confusion_matrix.png")

    summary = {
        "optimizer": opt_name, "seed": seed, "lr": lr,
        "epochs_run": len(hist), "best_epoch": best_ep,
        "total_time_s": round(total_time, 1),
        "s_per_epoch": round(total_time / len(hist), 2),
        "train_s_per_epoch": round(float(np.mean([h["train_time_s"] for h in hist])), 2),
        "train_acc": round(res["train"]["acc"], 4),
        "val_acc": round(res["validation"]["acc"], 4),
        "test_acc": round(te["acc"], 4),
        "val_macro_f1": round(res["validation"]["macro_f1"], 4),
        "test_macro_f1": round(report["macro avg"]["f1-score"], 4),
        "test_weighted_f1": round(report["weighted avg"]["f1-score"], 4),
    }
    for c in CLASSES:
        summary[f"test_f1_{c}"] = round(report[c]["f1-score"], 4)
    print(f"  ✓ best epoch {best_ep}  |  test acc {te['acc']:.3f}  "
          f"macro-F1 {summary['test_macro_f1']:.3f}  |  {total_time:.1f}s total")
    return summary, hist


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Compare optimizers on a fixed small CNN.")
    ap.add_argument("--data", help="path to strawberry_cls (default: auto-detect)")
    ap.add_argument("--optimizer", default="all", choices=OPTIMIZERS + ["all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None,
                    help="one learning rate for every optimizer (default: per-optimizer defaults)")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=10, help="early-stopping patience (epochs)")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    print("=" * 60 + "\n🍓 Optimizer comparison — small CNN\n" + "=" * 60)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        print(f"🖥  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("🖥  Running on CPU (no CUDA GPU detected)")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    root = find_data(args.data)
    print(f"📁 Data: {root}")
    arrays = load_arrays(root, args.img_size)

    # all data lives on the device as uint8; normalised per batch
    T = {s: (torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device),
             torch.from_numpy(y).to(device)) for s, (x, y) in arrays.items()}
    xf = T["train"][0].float() / 255
    mean = xf.mean(dim=(0, 2, 3), keepdim=True)
    std = xf.std(dim=(0, 2, 3), keepdim=True)
    del xf

    def prep(x):
        return (x.float() / 255 - mean) / std

    for s in SPLITS:
        counts = np.bincount(arrays[s][1], minlength=3)
        print(f"  {s:10s} {len(arrays[s][1]):5d} crops  "
              + "  ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))

    class_w = None
    if not args.no_class_weights:
        counts = np.bincount(arrays["train"][1], minlength=3).astype(np.float64)
        class_w = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)
        print("⚖️  class weights: " + ", ".join(f"{c} {w:.2f}"
                                               for c, w in zip(CLASSES, class_w.tolist())))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = root.parent / "runs" / f"{stamp}_{args.tag}"
    session.mkdir(parents=True, exist_ok=True)
    print(f"💾 Results: {session}")

    warm_up(device, args.img_size)
    opts = OPTIMIZERS if args.optimizer == "all" else [args.optimizer]
    rows = []
    for seed in args.seeds:
        histories = {}
        for o in opts:
            summary, hist = run_one(o, seed, T, prep, class_w, args, device, session)
            rows.append(summary)
            histories[o] = hist
            with open(session / "results.csv", "w", newline="") as f:   # rewritten after each run
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        if len(histories) > 1:
            plot_compare(histories, seed, session / f"compare_seed{seed}.png")

    # ---- summary table ------------------------------------------------
    print("\n" + "=" * 60 + "\n📊 RESULTS\n" + "=" * 60)
    print(f"{'optimizer':9s} {'seed':>4s} {'best ep':>7s} {'s/epoch':>8s} {'total s':>8s} "
          f"{'train acc':>9s} {'val acc':>8s} {'test acc':>8s} {'test F1':>8s}")
    for r in rows:
        print(f"{r['optimizer']:9s} {r['seed']:4d} {r['best_epoch']:7d} {r['s_per_epoch']:8.2f} "
              f"{r['total_time_s']:8.1f} {r['train_acc']:9.3f} {r['val_acc']:8.3f} "
              f"{r['test_acc']:8.3f} {r['test_macro_f1']:8.3f}")
    print(f"\n✅ Saved to {session}")


if __name__ == "__main__":
    main()
