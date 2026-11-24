"""Plot the matched-resolution optimizer comparison from archived run results."""

import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OPTIMIZERS = ("SGD", "Adagrad", "RMSprop", "Adam")


def read_runs(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def plot_runs(ax, runs, metric, color, offset=0.0, label=None):
    positions = [idx + offset for idx in range(len(OPTIMIZERS))]
    values = [
        [float(row[metric]) for row in runs if row["optimizer"] == name.lower()]
        for name in OPTIMIZERS
    ]
    for position, sample in zip(positions, values):
        jitter = [0.035 * (idx - (len(sample) - 1) / 2) for idx in range(len(sample))]
        ax.scatter(
            [position + shift for shift in jitter], sample, s=20, color=color, alpha=0.35
        )
    ax.errorbar(
        positions, [mean(sample) for sample in values],
        yerr=[stdev(sample) for sample in values], fmt="o", color=color,
        capsize=4, linewidth=1.7, label=label,
    )


def main():
    small = read_runs(ROOT / "docs" / "results" / "smallcnn-224" / "results.csv")
    resnet = read_runs(ROOT / "docs" / "results" / "resnet18" / "final" / "results.csv")
    fig, (score, epochs) = plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained")

    plot_runs(score, small, "test_macro_f1", "#8F3055", -0.13, "Small CNN")
    plot_runs(score, resnet, "test_macro_f1", "#137075", 0.13, "Pretrained ResNet-18")
    score.set(title="Held-out classification performance", ylabel="Test macro-F1", ylim=(0.82, 1.01))
    score.legend(loc="lower right", frameon=False)

    plot_runs(epochs, small, "epochs_to_target", "#8F3055")
    epochs.set(
        title="Small CNN convergence", ylabel="Epochs to validation macro-F1 ≥ 0.85",
        ylim=(0, 27),
    )

    for ax in (score, epochs):
        ax.set_xticks(range(len(OPTIMIZERS)), OPTIMIZERS)
        ax.set_xlim(-0.48, 3.48)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E3E6EA")
        ax.set_axisbelow(True)

    fig.suptitle("Strawberry leaf disease classification at 224 × 224", fontweight="bold")
    fig.supxlabel("Small dots show individual runs. Large dots and bars show mean ± SD (five seeds).", fontsize=9)
    out = ROOT / "docs" / "figures" / "optimizer_results_224.png"
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
