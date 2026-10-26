"""
Statistics, tables and confusion matrices for the final optimizer runs.

    python scripts/analyze_optimizers.py                 # reads runs/final/
    python scripts/analyze_optimizers.py runs/final

Decision rule (fixed before the experiment; follows the proposal's three criteria in order):
  1. Accuracy     - test macro-F1. If one optimizer is significantly better than every other
                    (paired t-test across seeds, Holm-corrected, alpha = 0.05), it is the best.
  2. Convergence  - otherwise, among the optimizers NOT significantly worse than the leader,
                    the one that reaches the target validation macro-F1 in significantly fewer
                    epochs is the best. Runs that never reach the target count as budget + 1.
  3. Training time- if still tied, the same test on wall-clock time to reach the target.
  If all three are tied, the honest conclusion is that the tied optimizers are equivalent.

Writes to <run_dir>/analysis/:
  summary.csv, significance.csv, per_class_metrics.csv, decision.txt,
  Figure_4_confusion_matrices.png/.pdf
Then draw Figures 1-3 with:  python scripts/legacy/make_figures_mdpi.py runs/final/results.csv runs/final/history.csv
"""
import sys, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
HERE = Path(__file__).resolve().parents[1]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs" / "final"
OUT = RUN / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05

ORDER_ALL = ["sgd", "adagrad", "rmsprop", "adam"]
LABEL = {"sgd": "SGD", "adagrad": "Adagrad", "rmsprop": "RMSprop", "adam": "Adam"}
PRETTY = {"angular_leaf_spot": "Angular leaf spot", "leaf_spot": "Leaf spot",
          "powdery_mildew_leaf": "Powdery mildew"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "TeX Gyre Heros", "DejaVu Sans"],
    "font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in", "figure.facecolor": "white",
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})

res = pd.read_csv(RUN / "results.csv")
pred = pd.read_csv(RUN / "predictions.csv")
ORDER = [o for o in ORDER_ALL if o in set(res.optimizer)]
classes = [c[len("test_f1_"):] for c in res.columns if c.startswith("test_f1_")]
names = [PRETTY.get(c, c.replace("_", " ").capitalize()) for c in classes]
budget = int(res.epoch_budget.max())
res["epochs_to_target_c"] = res.epochs_to_target.fillna(budget + 1)          # censored = never reached
res["time_to_target_c"] = res.time_to_target_s.fillna(res.total_time_s)


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def vs_leader(metric, opts, higher_is_better):
    """Paired (by seed) t-tests of the leader against every other optimizer in opts."""
    piv = res.pivot(index="seed", columns="optimizer", values=metric)[opts].dropna()
    means = piv.mean()
    leader = means.idxmax() if higher_is_better else means.idxmin()
    others = [o for o in opts if o != leader]
    raw = []
    for o in others:
        d = piv[leader] - piv[o]
        raw.append(1.0 if np.allclose(d, 0) else stats.ttest_rel(piv[leader], piv[o]).pvalue)
    adj = holm(raw) if raw else []
    rows = [dict(metric=metric, leader=LABEL[leader], other=LABEL[o], n_seeds=len(piv),
                 leader_mean=means[leader], other_mean=means[o], diff=means[leader] - means[o],
                 p_raw=r, p_holm=a, significant=a < ALPHA) for o, r, a in zip(others, raw, adj)]
    tied = [leader] + [o for o, a in zip(others, adj) if a >= ALPHA]
    return leader, tied, rows


def mean_sd(x, dec=3):
    return f"{np.mean(x):.{dec}f} ± {np.std(x, ddof=1):.{dec}f}"


# ── summary table ────────────────────────────────────────────────────────────
rows = []
for o in ORDER:
    d = res[res.optimizer == o]
    ci = stats.t.ppf(0.975, len(d) - 1) * d.test_macro_f1.std(ddof=1) / np.sqrt(len(d))
    rows.append({"Optimizer": LABEL[o], "LR": f"{d.lr.iloc[0]:g}", "Seeds": len(d),
                 "Test macro-F1": mean_sd(d.test_macro_f1),
                 "95% CI": f"[{d.test_macro_f1.mean() - ci:.3f}, {d.test_macro_f1.mean() + ci:.3f}]",
                 "Test accuracy": mean_sd(d.test_acc),
                 f"Epochs to val F1 ≥ {d.target_f1.iloc[0]:.2f}": mean_sd(d.epochs_to_target_c, 1)
                 + (f" ({d.epochs_to_target.isna().sum()} never)" if d.epochs_to_target.isna().any() else ""),
                 "Time to target (s, median)": f"{d.time_to_target_c.median():.0f}",
                 "s/epoch (median)": f"{d.s_per_epoch.median():.2f}",
                 "Best epoch": mean_sd(d.best_epoch, 1)})
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "summary.csv", index=False)

# ── decision rule ────────────────────────────────────────────────────────────
sig_rows, log = [], []
fried = stats.friedmanchisquare(*[res.pivot(index="seed", columns="optimizer", values="test_macro_f1")[o].dropna()
                                  for o in ORDER]) if len(ORDER) >= 3 else None

lead1, tied1, r1 = vs_leader("test_macro_f1", ORDER, True); sig_rows += r1
log.append(f"Criterion 1 (test macro-F1): leader {LABEL[lead1]}; "
           f"not significantly worse: {', '.join(LABEL[o] for o in tied1[1:]) or 'none'}")
winner, reason = None, ""
if len(tied1) == 1:
    winner, reason = lead1, "significantly highest test macro-F1"
else:
    lead2, tied2, r2 = vs_leader("epochs_to_target_c", tied1, False); sig_rows += r2
    log.append(f"Criterion 2 (epochs to target, among {', '.join(LABEL[o] for o in tied1)}): leader {LABEL[lead2]}; "
               f"not significantly slower: {', '.join(LABEL[o] for o in tied2[1:]) or 'none'}")
    if len(tied2) == 1:
        winner, reason = lead2, "statistically tied on accuracy, but reached the target in significantly fewer epochs"
    else:
        lead3, tied3, r3 = vs_leader("time_to_target_c", tied2, False); sig_rows += r3
        log.append(f"Criterion 3 (time to target, among {', '.join(LABEL[o] for o in tied2)}): leader {LABEL[lead3]}; "
                   f"not significantly slower: {', '.join(LABEL[o] for o in tied3[1:]) or 'none'}")
        if len(tied3) == 1:
            winner, reason = lead3, "tied on accuracy and epochs, but significantly less training time to the target"
        else:
            reason = (f"{', '.join(LABEL[o] for o in tied3)} are statistically equivalent on all three criteria; "
                      f"{LABEL[lead1]} has the highest mean macro-F1")
pd.DataFrame(sig_rows).to_csv(OUT / "significance.csv", index=False)

# ── per-class precision / recall / F1 (proposal deliverable 3) ──────────────
pc = []
for o in ORDER:
    per_seed = []
    for s, g in pred[pred.optimizer == o].groupby("seed"):
        t, p = g.y_true.to_numpy(), g.y_pred.to_numpy()
        pr, rc, f, sup = precision_recall_fscore_support(t, p, labels=list(range(len(classes))), zero_division=0)
        m = precision_recall_fscore_support(t, p, average="macro", zero_division=0)[:3]
        w = precision_recall_fscore_support(t, p, average="weighted", zero_division=0)[:3]
        per_seed.append(np.vstack([np.c_[pr, rc, f], m, w]))
    arr = np.stack(per_seed)                       # seeds x rows x (P, R, F1)
    for i, row in enumerate(names + ["Macro avg.", "Weighted avg."]):
        pc.append({"Optimizer": LABEL[o], "Class": row,
                   **{k: mean_sd(arr[:, i, j]) for j, k in enumerate(["Precision", "Recall", "F1 score"])}})
pd.DataFrame(pc).to_csv(OUT / "per_class_metrics.csv", index=False)

# ── confusion matrices (proposal deliverable 2) ─────────────────────────────
n = len(ORDER); ncol = 2 if n > 2 else n; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.3 * nrow), squeeze=False)
fig.subplots_adjust(wspace=0.45, hspace=0.55)
for k, (ax, o) in enumerate(zip(axes.flat, ORDER)):
    g = pred[pred.optimizer == o]
    cm = confusion_matrix(g.y_true, g.y_pred, labels=list(range(len(classes))))
    pct = cm / cm.sum(1, keepdims=True).clip(min=1) * 100
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{pct[i, j]:.1f}%\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if pct[i, j] > 55 else "black")
    short = [nm.replace(" ", "\n", 1) if len(nm) > 10 else nm for nm in names]
    ax.set_xticks(range(len(classes)), short, fontsize=8); ax.set_yticks(range(len(classes)), short, fontsize=8)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.text(0.5, -0.36, f"({'abcdefgh'[k]}) {LABEL[o]}", transform=ax.transAxes, ha="center", va="top")
for ax in axes.flat[n:]:
    ax.axis("off")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"Figure_4_confusion_matrices.{ext}")
plt.close(fig)

# ── report ───────────────────────────────────────────────────────────────────
lines = ["OPTIMIZER COMPARISON: DECISION", "=" * 60,
         f"Seeds per optimizer: {res.groupby('optimizer').seed.nunique().to_dict()}",
         f"Friedman test across all optimizers (test macro-F1): "
         f"chi2 = {fried.statistic:.2f}, p = {fried.pvalue:.4f}" if fried is not None else "",
         *log, "",
         f"BEST OPTIMIZER: {LABEL[winner]}  ({reason})" if winner else f"NO SINGLE BEST OPTIMIZER: {reason}", ""]
if winner:
    d = res[res.optimizer == winner]
    lines.append("Suggested results sentence:")
    lines.append(f"  {LABEL[winner]} (lr = {d.lr.iloc[0]:g}) was selected as the best optimizer: {reason} "
                 f"(test macro-F1 {mean_sd(d.test_macro_f1)}, n = {len(d)} seeds; paired t-tests, Holm-corrected).")
text = "\n".join(l for l in lines if l is not None)
(OUT / "decision.txt").write_text(text)
print(summary.to_string(index=False)); print(); print(text)
print(f"\nwritten to {OUT.resolve()}")
