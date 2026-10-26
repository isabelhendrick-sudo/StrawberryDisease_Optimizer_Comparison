"""Figures for the optimizer comparison, in a plain journal style
(simple lines, full frame, Blues heatmaps with counts, no bars or hatching).

    python scripts/make_figures.py

Reads   runs/final/results.csv, history.csv, predictions.csv  (at the repository root)
Writes  figures/  : Figure_1 ... Figure_4, Figure_S1 as PNG (600 dpi) and PDF, plus All_figures.pdf

Figure 1  Training curves          -> convergence (criterion 2)
Figure 2  Test performance         -> accuracy (criterion 1)
Figure 3  Convergence and time     -> criteria 2 and 3, the decision rule's own measures
Figure 4  Confusion matrices       -> proposal deliverable, one per optimizer
Figure S1 Generalisation           -> supplementary
"""
import argparse, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
HERE = Path(__file__).resolve().parents[1]      # repository root

ORDER = ["sgd", "adagrad", "rmsprop", "adam"]
LABEL = {"sgd": "SGD", "adagrad": "Adagrad", "rmsprop": "RMSprop", "adam": "Adam"}
# Okabe-Ito colours (as recommended in Wilke, Fundamentals of Data Visualization): readable with colour-vision deficiency
COLOR = {"sgd": "#F0E442", "adagrad": "#56B4E9", "rmsprop": "#009E73", "adam": "#E69F00"}
EDGE = "#333333"          # dark outline so the light yellow stays visible on white
MARKER = {"sgd": "o", "adagrad": "s", "rmsprop": "D", "adam": "^"}
CLASS_NAME = {"angular_leaf_spot": "Angular leaf spot", "leaf_spot": "Leaf spot", "powdery_mildew_leaf": "Powdery mildew"}
CLASS_ABBR = {"angular_leaf_spot": "ALS", "leaf_spot": "LS", "powdery_mildew_leaf": "PM"}
SAVED = []


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.linewidth": 0.8, "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": True, "legend.framealpha": 1, "legend.edgecolor": "0.8",
        "axes.titlesize": 10, "axes.titlelocation": "left", "lines.linewidth": 1.5,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })


def save(fig, name, out, book):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    book.savefig(fig)
    plt.close(fig)
    SAVED.append(name)


def mean_sd_points(ax, values, x, color, marker, rng):
    """Every run as a small marker, plus the mean ± SD as a large marker with a dark error bar."""
    v = np.asarray(values, float)
    ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)) - 0.14, v, s=30, marker=marker,
               facecolor=color, edgecolor=EDGE, lw=0.6, alpha=0.85, zorder=3)
    ax.errorbar(x + 0.14, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, fmt="none",
                ecolor=EDGE, capsize=4, elinewidth=1.1, zorder=4)
    ax.scatter(x + 0.14, v.mean(), s=90, marker=marker, facecolor=color, edgecolor=EDGE, lw=1, zorder=5)


# ── Figure 1: training curves ─────────────────────────────────────────────────
def training(h, target, n_runs, out, book):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True)
    min_runs = n_runs // 2 + 1                          # show the mean only while most runs are still training
    last_epoch = int(h.epoch.max())
    panels = [("train_loss", "Training loss"), ("val_loss", "Validation loss"),
              ("train_acc", "Training accuracy"), ("val_macro_f1", "Validation macro-F1")]
    for ax, (col, ylabel), letter in zip(axes.flat, panels, "abcd"):
        for o in ORDER:
            g = h[h.optimizer == o].groupby("epoch")[col].agg(["mean", "count"])
            g = g[g["count"] >= min_runs]
            ax.plot(g.index, g["mean"], color=COLOR[o], lw=2, label=LABEL[o])
        ax.set_xlim(0, last_epoch)                      # epoch 0 sits exactly in the corner
        if "loss" in col:
            ax.set_ylim(bottom=0)                       # loss axes start at 0 too
        ax.set_ylabel(ylabel)
        ax.set_title(f"({letter})")
    axes[1, 1].axhline(target, color="0.4", ls="--", lw=1)
    axes[1, 1].text(last_epoch, target, f"target {target:.2f} ", ha="right", va="bottom",
                    fontsize=8.5, color="0.3")
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "Figure_1_training_curves", out, book)


# ── Figure 2: test performance ────────────────────────────────────────────────
def performance(df, classes, out, book):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    for i, o in enumerate(ORDER):
        v = df.loc[df.optimizer == o, "test_macro_f1"]
        mean_sd_points(ax, v, i, COLOR[o], MARKER[o], rng)
        ax.annotate(f"{v.mean():.3f}", (i + 0.14, v.mean()), xytext=(11, 0), textcoords="offset points",
                    va="center", fontsize=8.5)
    lr = df.groupby("optimizer").lr.first()
    ax.set_xticks(range(len(ORDER)), [f"{LABEL[o]}\nlr = {lr[o]:g}" for o in ORDER])
    ax.set_xlim(-0.5, len(ORDER) - 0.3)
    ax.margins(y=0.15)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("(a)")

    ax = axes[1]
    cols = [f"test_f1_{c}" for c in classes] + ["test_macro_f1"]
    m = df.groupby("optimizer")[cols].mean().loc[ORDER].to_numpy()
    s = df.groupby("optimizer")[cols].std().loc[ORDER].to_numpy()
    vmin = np.floor(m.min() * 10) / 10
    im = ax.imshow(m, cmap="Blues", vmin=vmin, vmax=1.0, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            dark = (m[i, j] - vmin) / (1 - vmin) > 0.6
            ax.text(j, i, f"{m[i, j]:.3f}\n± {s[i, j]:.3f}", ha="center", va="center", fontsize=8.5,
                    color="white" if dark else "black")
    ax.set_xticks(range(len(cols)), [CLASS_NAME.get(c, c) for c in classes] + ["Macro avg."])
    ax.set_yticks(range(len(ORDER)), [LABEL[o] for o in ORDER])
    ax.axvline(len(classes) - 0.5, color="white", lw=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Test F1 (mean of runs)")
    ax.set_title("(b)")
    fig.tight_layout()
    save(fig, "Figure_2_test_performance", out, book)


# ── Figure 3: convergence and training time (the decision criteria) ──────────
def efficiency(df, out, book):
    rng = np.random.default_rng(1)
    target = float(df.target_f1.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    med = df.groupby("optimizer").s_per_epoch.median()
    ticks = [f"{LABEL[o]}\n{med[o]:.1f} s/epoch" for o in ORDER]
    for ax, col, ylabel, letter in [(axes[0], "epochs_to_target", f"Epochs to validation macro-F1 ≥ {target:.2f}", "a"),
                                    (axes[1], "time_to_target_s", "Training time to target (s)", "b")]:
        for i, o in enumerate(ORDER):
            d = df[df.optimizer == o]
            ok = d[col].dropna()
            if len(ok):
                mean_sd_points(ax, ok, i, COLOR[o], MARKER[o], rng)
            n_miss = d[col].isna().sum()
            if n_miss:
                ax.annotate(f"{n_miss} run(s)\nnot reached", (i, 1.0), xycoords=("data", "axes fraction"),
                            xytext=(0, -4), textcoords="offset points", ha="center", va="top", fontsize=8, color="0.3")
        ax.set_xticks(range(len(ORDER)), ticks)
        ax.set_xlim(-0.5, len(ORDER) - 0.5)
        ax.margins(y=0.2)
        ax.set_ylabel(ylabel)
        ax.set_title(f"({letter})")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "Figure_3_convergence_and_time", out, book)


# ── Figure 4: confusion matrices (one representative run per optimizer) ──────
def confusion(df, pred, classes, out, book):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.4))
    k = len(classes)
    mats, titles = [], []
    for o in ORDER:
        d = df[df.optimizer == o].sort_values(["test_macro_f1", "seed"]).reset_index(drop=True)
        rep = d.iloc[len(d) // 2]                                   # run with the median test macro-F1
        g = pred[(pred.optimizer == o) & (pred.seed == rep.seed)]
        mats.append(confusion_matrix(g.y_true, g.y_pred, labels=list(range(k))))
        titles.append(f"({'abcd'[len(titles)]}) {LABEL[o]}  (seed {int(rep.seed)}, macro-F1 {rep.test_macro_f1:.3f})")
    abbr = [CLASS_ABBR.get(c, c) for c in classes]
    for ax, cm, title in zip(axes.flat, mats, titles):
        # colour = share of each true class (classes are imbalanced); text = count and share
        pct = cm / cm.sum(1, keepdims=True).clip(min=1)
        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0%})", ha="center", va="center", fontsize=10,
                        color="white" if pct[i, j] > 0.55 else "black")
        ax.set_xticks(range(k), abbr); ax.set_yticks(range(k), abbr)
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
        ax.set_title(title, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    cax = fig.add_axes([0.9, 0.25, 0.022, 0.5])
    fig.colorbar(im, cax=cax, format=lambda x, _: f"{x:.0%}").set_label("Share of the true class")
    save(fig, "Figure_4_confusion_matrices", out, book)


# ── Figure S1: generalisation (supplementary) ─────────────────────────────────
def generalisation(df, out, book):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    for k, o in enumerate(ORDER):
        y = df.loc[df.optimizer == o, ["train_acc", "val_acc", "test_acc"]]
        ax.errorbar(np.arange(3) + (k - 1.5) * 0.07, y.mean(), yerr=y.std(ddof=1), color=COLOR[o], lw=2,
                    marker=MARKER[o], ms=7, mec=EDGE, mew=0.7, ecolor=EDGE, capsize=3, elinewidth=0.9,
                    label=LABEL[o])
    ax.set_xticks([0, 1, 2], ["Train", "Validation", "Test"])
    ax.set_xlim(-0.35, 2.35)
    ax.margins(y=0.15)
    ax.set_ylabel("Accuracy")
    ax.legend(loc="best")
    ax.set_title("(a)")

    ax = axes[1]
    v, t = df.val_macro_f1, df.test_macro_f1
    lo = min(v.min(), t.min()) - 0.02; hi = max(v.max(), t.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=1)
    for o in ORDER:
        d = df[df.optimizer == o]
        ax.scatter(d.val_macro_f1, d.test_macro_f1, marker=MARKER[o], s=46, facecolor=COLOR[o], edgecolor=EDGE,
                   lw=0.7, label=LABEL[o], zorder=3)
    r = np.corrcoef(v, t)[0, 1]
    ax.text(0.04, 0.96, f"r = {r:.2f} (n = {len(df)} runs)", transform=ax.transAxes, va="top")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Validation macro-F1"); ax.set_ylabel("Test macro-F1")
    ax.legend(loc="lower right")
    ax.set_title("(b)")
    fig.tight_layout()
    save(fig, "Figure_S1_generalisation", out, book)


def check(df, h, pred):
    """Fail loudly on problems instead of drawing a misleading figure."""
    need = {"optimizer", "seed", "lr", "test_macro_f1", "val_macro_f1", "train_acc", "val_acc", "test_acc",
            "s_per_epoch", "epochs_to_target", "time_to_target_s", "epoch_budget", "target_f1"}
    missing = need - set(df.columns)
    assert not missing, f"results.csv is missing columns: {missing}"
    assert set(df.optimizer) == set(ORDER), f"expected optimizers {ORDER}, found {sorted(set(df.optimizer))}"
    assert not df.duplicated(["optimizer", "seed"]).any(), "duplicate optimizer/seed rows in results.csv"
    assert (df.groupby("optimizer").lr.nunique() == 1).all(), "an optimizer appears with more than one learning rate"
    runs = set(zip(df.optimizer, df.seed))
    assert runs == set(zip(h.optimizer, h.seed)), "results.csv and history.csv contain different runs"
    assert runs == set(zip(pred.optimizer, pred.seed)), "results.csv and predictions.csv contain different runs"
    n = df.groupby("optimizer").seed.nunique()
    assert n.nunique() == 1, f"optimizers have different numbers of runs: {n.to_dict()}"
    return int(n.iloc[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=HERE / "runs" / "final", help="folder with the final-run CSVs")
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    files = {k: args.runs / f"{k}.csv" for k in ("results", "history", "predictions")}
    for f in files.values():
        if not f.exists():
            raise SystemExit(f"Can't find {f}\nIt is created when 'python scripts/train_optimizers.py final' finishes.")
    df, h, pred = (pd.read_csv(files[k]) for k in ("results", "history", "predictions"))
    for frame in (df, h, pred):
        frame["optimizer"] = frame.optimizer.str.lower()
    n_runs = check(df, h, pred)
    classes = [c[len("test_f1_"):] for c in df.columns if c.startswith("test_f1_")]
    target = float(df.target_f1.iloc[0])

    style()
    args.out.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out / "All_figures.pdf") as book:
        training(h, target, n_runs, args.out, book)
        performance(df, classes, args.out, book)
        efficiency(df, args.out, book)
        confusion(df, pred, classes, args.out, book)
        generalisation(df, args.out, book)
    audit = {"runs_per_optimizer": n_runs, "total_runs": len(df), "all_runs_plotted": True,
             "sd": "sample SD (ddof = 1)", "smoothing": False,
             "mean_test_macro_f1": df.groupby("optimizer").test_macro_f1.mean().round(4).to_dict(),
             "median_s_per_epoch": df.groupby("optimizer").s_per_epoch.median().round(2).to_dict(),
             "figures": SAVED}
    (args.out / "data_checks.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"{len(SAVED)} figures ({n_runs} runs per optimizer) written to {args.out.resolve()}")


if __name__ == "__main__":
    main()