"""
Optimizer comparison for a small CNN on three strawberry leaf-disease classes.

Phase 1  sweep   Every optimizer gets the SAME learning-rate grid and the SAME budget.
                 The best LR per optimizer is chosen on VALIDATION macro-F1 only
                 (test data is never looked at), using a seed not used in phase 2.
Phase 2  final   Each optimizer at its own tuned LR, repeated over several seeds.
                 For a given seed every optimizer starts from identical weights and
                 sees batches in an identical order, so runs can be compared pairwise.

Models
  --model smallcnn   the 4-block CNN from the proposal (default)
  --model resnet18   ImageNet-pretrained ResNet-18; only layer4 and the new classifier are trained
                     (use with --img 224 and its own --lr-grid)

Usage
  python train_optimizers.py smoke    # 2-epoch sanity check (run this first, ~2 min)
  python train_optimizers.py sweep    # 4 optimizers x 6 LRs = 24 runs
  python train_optimizers.py final    # 4 optimizers x 5 seeds = 20 runs
  python analyze_optimizers.py        # statistics, tables, confusion matrices, decision

Data layout (strawberry_cls, next to this script):
  strawberry_cls/train/<class>/*.jpg
  strawberry_cls/validation/<class>/*.jpg
  strawberry_cls/test/<class>/*.jpg
  with class folders: angular_leaf_spot, leaf_spot, powdery_mildew_leaf

Runs that already finished are skipped, so you can stop and restart at any time.
"""
import argparse, json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, precision_recall_fscore_support

OPTIMIZERS = ["sgd", "adagrad", "rmsprop", "adam"]
HERE = Path(__file__).resolve().parent          # the folder this script is in


# ─────────────────────────────── settings ────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["smoke", "sweep", "final"])
    p.add_argument("--data", default=str(HERE / "strawberry_cls"), help="folder with train/ validation/ test/")
    p.add_argument("--model", choices=["smallcnn", "resnet18"], default="smallcnn")
    p.add_argument("--out", default=str(HERE / "runs"))
    p.add_argument("--img", type=int, default=128, help="input size (proposal: 128 x 128 x 3)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40, help="epoch budget, same for every run")
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience on val macro-F1")
    p.add_argument("--lr-grid", type=float, nargs="+", default=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4])
    p.add_argument("--sweep-seed", type=int, default=100, help="kept separate from the final seeds")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--target-f1", type=float, default=0.85,
                   help="'useful performance' for the convergence criterion; fix this BEFORE running")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--save-weights", action="store_true", help="keep best.pt for every final run")
    return p.parse_args()


# ─────────────────────────────── utilities ───────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    """Wait for the GPU so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ─────────────────────────────── data ────────────────────────────────────────
def load_split(root, split, img):
    """Load a whole split into memory as uint8 tensors (fast, and removes disk I/O from the timings)."""
    ds = datasets.ImageFolder(root / split, transform=transforms.Compose(
        [transforms.Resize((img, img)), transforms.PILToTensor()]))
    X = torch.stack([x for x, _ in ds])
    y = torch.tensor(ds.targets, dtype=torch.long)
    return X, y, ds.classes


def channel_stats(X, chunk=512):
    """Per-channel mean/std of the TRAINING set only (no information from val/test)."""
    s = torch.zeros(3, dtype=torch.float64); s2 = torch.zeros(3, dtype=torch.float64); n = 0
    for i in range(0, len(X), chunk):
        b = X[i:i + chunk].double().div_(255)
        s += b.sum((0, 2, 3)); s2 += (b * b).sum((0, 2, 3)); n += b.numel() // 3
    mean = s / n
    return mean.float(), (s2 / n - mean ** 2).sqrt().float()


IMAGENET_MEAN, IMAGENET_STD = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])


def load_data(root, img, device, model="smallcnn"):
    root = Path(root)
    if not root.is_absolute() and not root.exists():
        root = HERE / root                              # allow --data aihub_cls from any folder
    if not (root / "train").is_dir():
        raise SystemExit(f"No 'train' folder found in {root.resolve()}")
    val_name = next((n for n in ("val", "valid", "validation") if (root / n).is_dir()), None)
    if val_name is None:
        raise SystemExit(f"No 'val', 'valid' or 'validation' folder found in {root.resolve()}")
    Xtr, ytr, classes = load_split(root, "train", img)
    Xva, yva, c_va = load_split(root, val_name, img)
    Xte, yte, c_te = load_split(root, "test", img)
    assert classes == c_va == c_te, f"class folders differ between splits: {classes} {c_va} {c_te}"
    # the pretrained ResNet expects the ImageNet normalisation it was trained with
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if model == "resnet18" else channel_stats(Xtr)
    print(f"classes: {classes}")
    print(f"images  train {len(ytr)}  val {len(yva)}  test {len(yte)}")
    for name, y in [("train", ytr), ("val", yva), ("test", yte)]:
        print(f"  {name:5s} per class: {torch.bincount(y, minlength=len(classes)).tolist()}")
    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte, classes=classes,
                mean=mean.view(1, 3, 1, 1).to(device), std=std.view(1, 3, 1, 1).to(device))


def batches(X, y, bs, shuffle, gen=None):
    idx = torch.randperm(len(X), generator=gen) if shuffle else torch.arange(len(X))
    for i in range(0, len(X), bs):
        j = idx[i:i + bs]
        yield X[j], y[j]


def prep(xb, data, device):
    xb = xb.to(device, non_blocking=True).float().div_(255)
    return (xb - data["mean"]) / data["std"]


# ─────────────────────────────── model ───────────────────────────────────────
class SmallCNN(nn.Module):
    """4 conv blocks (16-32-64-128, 3x3) + global average pooling + dense 64 + 3-class output."""

    def __init__(self, n_classes=3):
        super().__init__()
        layers, c_in = [], 3
        for c_out in (16, 32, 64, 128):
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(inplace=True),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


class ResNet18FineTune(nn.Module):
    """ImageNet-pretrained ResNet-18 with a new classifier. Only layer4 and fc are trained;
    the frozen layers also keep their pretrained batch-norm statistics."""
    TRAINABLE = ("layer4", "fc")

    def __init__(self, n_classes=3):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(net.fc.in_features, n_classes)
        for name, prm in net.named_parameters():
            prm.requires_grad = name.startswith(self.TRAINABLE)
        self.net = net

    def train(self, mode=True):
        super().train(mode)
        for name, child in self.net.named_children():
            if not name.startswith(self.TRAINABLE):
                child.eval()                            # frozen part: never update batch-norm statistics
        return self

    def forward(self, x):
        return self.net(x)


def build_model(name, n_classes):
    return ResNet18FineTune(n_classes) if name == "resnet18" else SmallCNN(n_classes)


def make_optimizer(name, params, lr):
    # Library defaults for everything except the learning rate, so the optimizer is the only change.
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)          # plain SGD, no momentum (as in the proposal)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(name)


# ─────────────────────────────── train / evaluate ────────────────────────────
@torch.no_grad()
def evaluate(model, X, y, data, device, bs, loss_fn):
    model.eval()
    preds, loss_sum = [], torch.zeros((), device=device)
    for xb, yb in batches(X, y, bs, shuffle=False):
        out = model(prep(xb, data, device))
        loss_sum += loss_fn(out, yb.to(device)) * len(yb)
        preds.append(out.argmax(1).cpu())
    p, t = torch.cat(preds).numpy(), y.numpy()
    return dict(loss=loss_sum.item() / len(y), acc=float((p == t).mean()),
                macro_f1=float(f1_score(t, p, average="macro", zero_division=0)), pred=p)


def run(opt_name, lr, seed, data, args, device, run_dir, save_weights=False):
    """Train one model; write result.json, history.csv and predictions.csv into run_dir."""
    if (run_dir / "result.json").exists():
        print(f"skip (done): {run_dir.name}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)                                         # same seed -> same initial weights for all optimizers
    model = build_model(args.model, len(data["classes"])).to(device)
    opt = make_optimizer(opt_name, [q for q in model.parameters() if q.requires_grad], lr)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)              # same batch order for all optimizers

    hist, best_f1, best_epoch, best_state, bad = [], -1.0, 0, None, 0
    reached_epoch = reached_time = None
    diverged, elapsed = False, 0.0

    for epoch in range(1, args.epochs + 1):
        model.train(); sync(device); t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device); correct = torch.zeros((), device=device); n = 0
        for xb, yb in batches(data["Xtr"], data["ytr"], args.batch, shuffle=True, gen=gen):
            xb, yb = prep(xb, data, device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * len(yb)
            correct += (out.argmax(1) == yb).sum()
            n += len(yb)
        sync(device); t_train = time.perf_counter() - t0
        v = evaluate(model, data["Xva"], data["yva"], data, device, args.batch, loss_fn)
        sync(device); t_epoch = time.perf_counter() - t0
        elapsed += t_epoch

        train_loss = loss_sum.item() / n
        hist.append(dict(optimizer=opt_name, seed=seed, lr=lr, epoch=epoch,
                         train_loss=train_loss, train_acc=correct.item() / n,
                         val_loss=v["loss"], val_acc=v["acc"], val_macro_f1=v["macro_f1"],
                         epoch_s=t_epoch, train_s=t_train, elapsed_s=elapsed))
        print(f"[{opt_name:7s} lr={lr:<7g} seed={seed}] ep {epoch:02d}  "
              f"train loss {train_loss:.3f} acc {correct.item() / n:.3f}  |  "
              f"val loss {v['loss']:.3f} acc {v['acc']:.3f} F1 {v['macro_f1']:.3f}  ({t_epoch:.1f}s)")

        if reached_epoch is None and v["macro_f1"] >= args.target_f1:
            reached_epoch, reached_time = epoch, elapsed
        if v["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = v["macro_f1"], epoch, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if not math.isfinite(train_loss):
            diverged = True
            print("  loss is NaN/inf -> learning rate too high, stopping this run")
            break
        if bad >= args.patience:
            break

    # evaluate the best-validation checkpoint on val and test
    model.load_state_dict(best_state)
    v = evaluate(model, data["Xva"], data["yva"], data, device, args.batch, loss_fn)
    te = evaluate(model, data["Xte"], data["yte"], data, device, args.batch, loss_fn)
    t, p = data["yte"].numpy(), te["pred"]
    k = len(data["classes"])
    prec, rec, f1, _ = precision_recall_fscore_support(t, p, labels=list(range(k)), zero_division=0)

    res = dict(optimizer=opt_name, seed=int(seed), lr=float(lr), model=args.model, img=args.img,
               epoch_budget=args.epochs,
               epochs_run=len(hist), best_epoch=int(best_epoch),
               total_time_s=round(elapsed, 2),
               s_per_epoch=round(float(np.mean([h["epoch_s"] for h in hist])), 3),
               train_s_per_epoch=round(float(np.mean([h["train_s"] for h in hist])), 3),
               train_acc=hist[best_epoch - 1]["train_acc"], val_acc=v["acc"], test_acc=te["acc"],
               val_macro_f1=v["macro_f1"], test_macro_f1=te["macro_f1"],
               test_weighted_f1=float(f1_score(t, p, average="weighted", zero_division=0)),
               target_f1=args.target_f1,
               epochs_to_target=reached_epoch if reached_epoch is not None else float("nan"),
               time_to_target_s=round(reached_time, 2) if reached_time is not None else float("nan"),
               diverged=diverged, device=device.type)
    for i, c in enumerate(data["classes"]):
        res[f"test_f1_{c}"] = float(f1[i])
        res[f"test_precision_{c}"] = float(prec[i])
        res[f"test_recall_{c}"] = float(rec[i])

    pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame(dict(optimizer=opt_name, seed=seed, lr=lr, y_true=t, y_pred=p,
                      true_label=[data["classes"][i] for i in t],
                      pred_label=[data["classes"][i] for i in p])).to_csv(run_dir / "predictions.csv", index=False)
    if save_weights:
        torch.save(best_state, run_dir / "best.pt")
    (run_dir / "result.json").write_text(json.dumps(res, indent=2))   # written last = run is complete
    print(f"  -> best epoch {best_epoch}, val F1 {v['macro_f1']:.3f}, test F1 {te['macro_f1']:.3f}\n")


def collect(phase_dir):
    """Merge every finished run in phase_dir into results.csv, history.csv, predictions.csv."""
    dirs = sorted(d for d in phase_dir.iterdir() if (d / "result.json").exists())
    res = pd.DataFrame([json.loads((d / "result.json").read_text()) for d in dirs])
    res.to_csv(phase_dir / "results.csv", index=False)
    pd.concat([pd.read_csv(d / "history.csv") for d in dirs]).to_csv(phase_dir / "history.csv", index=False)
    pd.concat([pd.read_csv(d / "predictions.csv") for d in dirs]).to_csv(phase_dir / "predictions.csv", index=False)
    return res


def run_name(o, lr, s):
    return f"{o}_lr{lr:g}_seed{s}"


# ─────────────────────────────── main ────────────────────────────────────────
def main():
    args = parse_args()
    device = pick_device(args.device)
    out = Path(args.out)
    print(f"device: {device}  model: {args.model}  input: {args.img}x{args.img}")
    if args.model == "resnet18" and args.img < 224:
        print("NOTE: the pretrained ResNet-18 is normally used with --img 224")
    data = load_data(args.data, args.img, device, args.model)

    if args.phase == "smoke":
        args.epochs, args.patience = 2, 2
        d = out / "smoke"
        for o in OPTIMIZERS:
            run(o, 1e-2, 0, data, args, device, d / run_name(o, 1e-2, 0))
        print(collect(d)[["optimizer", "epochs_run", "val_macro_f1", "test_macro_f1", "s_per_epoch"]])
        print("\nSmoke test passed. Delete runs/smoke before the real runs if you like.")

    elif args.phase == "sweep":
        d = out / "sweep"
        for lr in args.lr_grid:                       # interleave optimizers so background load hits all equally
            for o in OPTIMIZERS:
                run(o, lr, args.sweep_seed, data, args, device, d / run_name(o, lr, args.sweep_seed))
        res = collect(d)
        best = res.loc[res.groupby("optimizer").val_macro_f1.idxmax()].set_index("optimizer")
        chosen = {o: float(best.loc[o, "lr"]) for o in OPTIMIZERS}
        (d / "best_lr.json").write_text(json.dumps(chosen, indent=2))
        table = res.pivot(index="lr", columns="optimizer", values="val_macro_f1")[OPTIMIZERS]
        print("\nValidation macro-F1 by learning rate:\n", table.round(3).sort_index(ascending=False))
        print("\nChosen learning rates:", chosen)
        lo, hi = min(args.lr_grid), max(args.lr_grid)
        for o, lr in chosen.items():
            if lr in (lo, hi):
                print(f"  NOTE: {o} picked the edge of the grid ({lr:g}); consider extending --lr-grid.")

    elif args.phase == "final":
        lr_file = out / "sweep" / "best_lr.json"
        if not lr_file.exists():
            raise SystemExit("Run the sweep first:  python train_optimizers.py sweep")
        chosen = json.loads(lr_file.read_text())
        print("using tuned learning rates:", chosen)
        d = out / "final"
        for s in args.seeds:                          # interleave optimizers within each seed
            for o in OPTIMIZERS:
                run(o, chosen[o], s, data, args, device, d / run_name(o, chosen[o], s), args.save_weights)
        res = collect(d)
        print(res.groupby("optimizer")[["test_macro_f1", "test_acc", "epochs_to_target", "s_per_epoch"]]
              .agg(["mean", "std"]).round(3))
        print(f"\nwritten: {d / 'results.csv'}, history.csv, predictions.csv")
        print("next:    python analyze_optimizers.py")


if __name__ == "__main__":
    main()
