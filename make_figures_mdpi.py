"""MDPI-style figures for the optimizer comparison (strawberry leaf disease).

Usage (from anywhere):
    python scripts/legacy/make_figures_mdpi.py

Reads  runs/final/results.csv  and  runs/final/history.csv  (at the repository root)
Writes figures_mdpi/  (at the repository root)
"""
import sys, logging, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

HERE = Path(__file__).resolve().parents[2]      # repository root
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs" / "final" / "results.csv"
HISTORY = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS.parent / "history.csv"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "figures_mdpi"
if not RESULTS.exists():
    raise SystemExit(f"Can't find {RESULTS}\n"
                     "It is created when 'python scripts/train_optimizers.py final' finishes all runs.")
if not HISTORY.exists():
    print(f"note: {HISTORY} not found, skipping the training-curve figure")
    HISTORY = None
OUT.mkdir(parents=True, exist_ok=True)

# ── MDPI journal look: Arial/Helvetica, full black frame, inward ticks, no titles ──
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "TeX Gyre Heros", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.linewidth": 0.8, "axes.edgecolor": "black", "axes.labelcolor": "black", "text.color": "black",
    "xtick.color": "black", "ytick.color": "black", "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.visible": False, "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False, "legend.framealpha": 1,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "lines.linewidth": 1.4,
})

ORDER = ["sgd", "adagrad", "rmsprop", "adam"]
LABEL = {"sgd": "SGD", "adagrad": "Adagrad", "rmsprop": "RMSprop", "adam": "Adam"}
COL = {"sgd": "#0072B2", "adagrad": "#D55E00", "rmsprop": "#009E73", "adam": "#CC79A7"}   # Okabe–Ito, colour-blind safe
MARK = {"sgd": "o", "adagrad": "s", "rmsprop": "D", "adam": "^"}
HATCH = {"sgd": "", "adagrad": "///", "rmsprop": "...", "adam": "xxx"}                     # survives greyscale printing

def panel(ax, letter, y=-0.2):
    """MDPI convention: sub-figure label centred beneath each panel."""
    ax.text(0.5, y, f"({letter})", transform=ax.transAxes, ha="center", va="top", fontsize=10)

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)

df = pd.read_csv(RESULTS)
df["optimizer"] = df.optimizer.str.lower()
LR = df.groupby("optimizer").lr.first()
tick = [f"{LABEL[o]}\n{LR[o]:g}" for o in ORDER]
rng = np.random.default_rng(0)

# ── Figure 1: training curves (needs per-epoch history) ──────────────────────
if HISTORY is not None:
    h = pd.read_csv(HISTORY); h["optimizer"] = h.optimizer.str.lower()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0)); fig.subplots_adjust(wspace=0.28)
    for ax, col, ylab in zip(axes, ["val_loss", "val_macro_f1"], ["Validation loss", "Validation macro-F1"]):
        for o in ORDER:
            d = h[h.optimizer == o]
            g = d.groupby("epoch")[col].agg(["mean", "std", "count"])
            g = g[g["count"] >= 2]                        # stop the mean where fewer than 2 seeds are still training
            ax.plot(g.index, g["mean"], color=COL[o], marker=MARK[o], ms=3.5, markevery=3, label=LABEL[o])
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=COL[o], alpha=0.15, lw=0)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylab); ax.set_xlim(0, h.epoch.max() + 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="lower right")
    panel(axes[0], "a"); panel(axes[1], "b")
    save(fig, "Figure_1_training_curves")

# ── Figure 2: overall and per-class test performance (the headline result) ──
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1, 1.35])); fig.subplots_adjust(wspace=0.3)
ax = axes[0]
for i, o in enumerate(ORDER):
    v = df.loc[df.optimizer == o, "test_macro_f1"].to_numpy()
    ax.bar(i, v.mean(), 0.62, color=COL[o], alpha=0.85, edgecolor="black", lw=0.6, hatch=HATCH[o],
           yerr=v.std(ddof=1), error_kw=dict(ecolor="black", lw=0.8, capsize=3))
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color="white", edgecolor="black", lw=0.7, zorder=3)
    ax.text(i, v.mean() + v.std(ddof=1) + 0.012, f"{v.mean():.3f}", ha="center", fontsize=8.5)
ax.set_xticks(range(4), tick, fontsize=9); ax.set_ylabel("Test macro-F1"); ax.set_ylim(0.6, 1.0)
ax.set_xlabel("Optimizer (learning rate)")
ax.tick_params(axis="x", top=False)

cls = [c for c in df.columns if c.startswith("test_f1_")]
pretty = {"angular_leaf_spot": "Angular leaf spot", "leaf_spot": "Leaf spot", "powdery_mildew_leaf": "Powdery mildew"}
names = [pretty.get(c[8:], c[8:].replace("_", " ").capitalize()) for c in cls]
m = df.groupby("optimizer")[cls].mean().loc[ORDER]; s = df.groupby("optimizer")[cls].std().loc[ORDER]
ax = axes[1]; w = 0.2
for k, o in enumerate(ORDER):
    ax.bar(np.arange(len(cls)) + (k - 1.5) * w, m.loc[o], w, color=COL[o], alpha=0.85, edgecolor="black", lw=0.6,
           hatch=HATCH[o], yerr=s.loc[o], error_kw=dict(ecolor="black", lw=0.7, capsize=2), label=LABEL[o])
ax.set_xticks(range(len(cls)), names); ax.set_ylabel("Test F1-score"); ax.set_ylim(0.5, 1.12)
ax.set_xlabel("Disease class"); ax.set_yticks(np.arange(0.5, 1.01, 0.1))
ax.tick_params(axis="x", top=False)
ax.legend(ncol=4, loc="upper center", fontsize=8, handlelength=1.4, columnspacing=0.9, borderpad=0.3)
panel(axes[0], "a", -0.26); panel(axes[1], "b", -0.26)
save(fig, "Figure_2_test_performance")

# ── Figure 3: training efficiency ───────────────────────────────────────────
med = df.s_per_epoch.median()
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0)); fig.subplots_adjust(wspace=0.3)
for i, o in enumerate(ORDER):
    d = df[df.optimizer == o]
    for _, r in d.iterrows():
        slow = r.s_per_epoch > 2 * med
        kw = dict(marker=MARK[o], s=30, lw=0.9, zorder=3, edgecolor=COL[o] if slow else "black",
                  facecolor="none" if slow else COL[o])
        axes[0].scatter(i + rng.uniform(-0.12, 0.12), r.s_per_epoch, **kw)
        axes[1].scatter(r.best_epoch, r.test_macro_f1, **kw)
    axes[1].scatter([], [], marker=MARK[o], s=30, color=COL[o], edgecolor="black", lw=0.6, label=LABEL[o])
axes[0].axhline(med, color="black", ls="--", lw=0.8)
axes[0].set_xticks(range(4), [LABEL[o] for o in ORDER]); axes[0].tick_params(axis="x", top=False)
axes[0].set_ylabel("Time per epoch (s)"); axes[0].set_xlim(-0.6, 3.6)
axes[1].set_xlabel("Best epoch (validation)"); axes[1].set_ylabel("Test macro-F1")
axes[1].legend(loc="lower right")
panel(axes[0], "a"); panel(axes[1], "b")
save(fig, "Figure_3_efficiency")

# ── Figure S1 (supplementary): generalisation ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0)); fig.subplots_adjust(wspace=0.3)
for k, o in enumerate(ORDER):
    y = df.loc[df.optimizer == o, ["train_acc", "val_acc", "test_acc"]]
    axes[0].errorbar(np.arange(3) + (k - 1.5) * 0.06, y.mean(), yerr=y.std(), color=COL[o], marker=MARK[o], ms=5, mec="black", mew=0.5,
                     capsize=2.5, elinewidth=0.7, label=LABEL[o])
    d = df[df.optimizer == o]
    axes[1].scatter(d.val_macro_f1, d.test_macro_f1, marker=MARK[o], s=30, color=COL[o], edgecolor="black", lw=0.6, label=LABEL[o])
axes[0].set_xticks([0, 1, 2], ["Train", "Validation", "Test"]); axes[0].set_ylabel("Accuracy"); axes[0].set_xlim(-0.3, 2.3)
axes[0].set_ylim(0.74, 1.0); axes[0].legend(loc="lower center", ncol=2, fontsize=8)
lo, hi = 0.75, 0.95
axes[1].plot([lo, hi], [lo, hi], color="black", ls="--", lw=0.8)
r = np.corrcoef(df.val_macro_f1, df.test_macro_f1)[0, 1]
axes[1].text(0.05, 0.93, f"r = {r:.2f}", transform=axes[1].transAxes, va="top")
axes[1].set_xlim(lo, hi); axes[1].set_ylim(lo, hi); axes[1].set_aspect("equal")
axes[1].set_xlabel("Validation macro-F1"); axes[1].set_ylabel("Test macro-F1")
panel(axes[0], "a"); panel(axes[1], "b")
save(fig, "Figure_S1_generalisation")

# ── Table: mean ± SD per optimizer + significance vs. best ──────────────────
cols = {"test_acc": "Test accuracy", "test_macro_f1": "Test macro-F1", "test_weighted_f1": "Test weighted-F1",
        "best_epoch": "Best epoch", "s_per_epoch": "Time/epoch (s, median)"}
best = df.groupby("optimizer").test_macro_f1.mean().idxmax()
rows = []
for o in ORDER:
    d = df[df.optimizer == o]; row = {"Optimizer": LABEL[o], "lr": LR[o]}
    for c, n in cols.items():
        dec = 1 if c in ("best_epoch",) else 2 if c == "s_per_epoch" else 3
        row[n] = f"{d[c].median():.2f}" if c == "s_per_epoch" else f"{d[c].mean():.{dec}f} ± {d[c].std():.{dec}f}"
    row[f"p vs {LABEL[best]} (paired t, macro-F1)"] = "–" if o == best else \
        f"{stats.ttest_rel(df[df.optimizer == best].sort_values('seed').test_macro_f1, d.sort_values('seed').test_macro_f1).pvalue:.3f}"
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "Table_optimizer_summary.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print("written to", OUT.resolve())