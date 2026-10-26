# Strawberry Leaf Disease: Optimizer Comparison

Comparison of **SGD, Adagrad, RMSprop, and Adam** for classifying strawberry leaf crops into **angular leaf spot, leaf spot, and powdery mildew leaf**.

The study follows a pilot experiment, a controlled learning-rate sweep and five-seed comparison on a small CNN, then a pretrained ResNet-18 comparison. **Adam has the highest small-CNN mean test macro-F1 (0.907), but the decision rule does not identify a single best optimizer among Adam, RMSprop, and Adagrad.** The presentation reports macro-F1 of approximately **0.99** for every optimizer with ResNet-18.

The research order and supporting discussion follow [the final presentation](docs/presentation/Strawberry_Optimizer_Final_2.pptx). Small-CNN tables below are recalculated from the repository's [20-run CSV](docs/results/smallcnn/results.csv). Pilot, dataset-cleaning, sweep, and ResNet-18 values come from the presentation where raw records are unavailable. See [source provenance](docs/PROVENANCE.md).

## Contents

- [Research question](#research-question)
- [Dataset and cleaning](#dataset-and-cleaning)
- [Methodology](#methodology)
- [Pilot and revised design](#pilot-and-revised-design)
- [Learning-rate sweep](#learning-rate-sweep)
- [Small-CNN results](#small-cnn-results)
- [ResNet-18 comparison](#resnet-18-comparison)
- [Discussion and conclusion](#discussion-and-conclusion)
- [Repository structure](#repository-structure)
- [Setup and usage](#setup-and-usage)

## Research question

How does optimizer choice affect **classification performance**, **convergence speed**, and **training time** when the CNN architecture and other training conditions are held fixed?

The project produces learning curves, confusion matrices, per-disease precision/recall/F1, and timing summaries. Macro-F1 is the primary classification metric because it weights the three diseases equally, despite their unequal representation.

## Dataset and cleaning

### Source dataset

The presentation identifies **Strawberry Disease Detection Dataset v4**, exported from Roboflow Universe in YOLOv8 format: **4,918 images**, resized to **280 × 280** with letterboxing, with no export-time augmentation. It reports a **CC BY 4.0** dataset license. The exact Roboflow workspace/version URL and attribution details are not included in the supplied materials and should be recorded with the dataset before redistribution.

The original export covers seven classes and contains 3,440 training, 992 validation, and 486 test images. This study retains original class IDs **0: angular leaf spot**, **4: leaf spot**, and **6: powdery mildew leaf**. The source detection labels are converted into cropped-leaf classification examples.

### Cleaning and grouping

| Cleaning step | Photos / crops |
| --- | ---: |
| Exported images reported by the dataset | 4,918 |
| Photos scanned by the cleaning script | 4,898 |
| Removed: no selected disease class | 1,828 |
| Removed: duplicate photos | 1,823 |
| Unique photos retained | 1,247 |
| Small boxes excluded | 55 |
| Final leaf crops | 3,429 |

The 20-image difference between the reported export and scan counts is retained from slide 5; the supplied records do not explain it.

Cleaning removes near duplicates using perceptual hashes, including flipped and rotated views, before splitting. The script uses a 64-bit hash-distance threshold of 6, trims black letterbox edges, adds a 15% crop margin, and rejects source boxes smaller than 24 pixels. It splits by source photo with seed 42, so crops from the same retained photo stay together.

| Split | Angular leaf spot | Leaf spot | Powdery mildew leaf | Total crops | Source photos |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 269 | 968 | 1,131 | 2,368 | 872 |
| Validation | 49 | 210 | 264 | 523 | 187 |
| Test | 58 | 205 | 275 | 538 | 188 |
| **Total** | **376** | **1,383** | **1,670** | **3,429** | **1,247** |

The approximate **70/15/15 split applies to photos**, not independently to crops. Angular leaf spot contributes about 11% of all crops and only 58 test examples.

## Methodology

### Small-CNN architecture

```text
RGB input: 128 × 128 × 3
  → Conv 3×3 (16 channels)  → ReLU → MaxPool 2×2
  → Conv 3×3 (32 channels)  → ReLU → MaxPool 2×2
  → Conv 3×3 (64 channels)  → ReLU → MaxPool 2×2
  → Conv 3×3 (128 channels) → ReLU → MaxPool 2×2
  → Global average pooling
  → Dense 64 → ReLU
  → Three class logits
```

The final trainer uses training-set channel normalization, plain cross-entropy loss, batch size 32, no dropout, and no training augmentation. Each run has a maximum of 40 epochs and early-stopping patience of 10 on validation macro-F1. The checkpoint with the highest validation macro-F1 is evaluated on the test set.

Every optimizer receives the same learning-rate grid and epoch budget. Final seeds **0–4** are paired: each optimizer starts from the same weights and sees the same batch order for a given seed. The tuning seed is **100**. SGD has no momentum; other optimizer parameters retain their library defaults.

### Optimizer selection rule

1. Compare test macro-F1 using paired t-tests across seeds, with Holm correction at α = 0.05.
2. If the accuracy comparisons leave several candidates, compare epochs needed to reach **validation macro-F1 ≥ 0.85**.
3. If still unresolved, compare wall-clock time to the same target.

Report no unique winner if the tests do not separate the remaining candidates. Nonsignificant differences are not evidence of statistical equivalence. A Friedman test provides a separate overall comparison among the four optimizers.

The code selects learning rates using validation macro-F1 only. It nevertheless logs test metrics during the sweep, so the presentation's statement that the test set is “never used” should be read as **not used for learning-rate selection**, rather than never evaluated.

## Pilot and revised design

The pilot used three seeds, fixed default learning rates, dropout 0.3, and class-weighted loss. These rounded values are from slide 24, not the final-run CSV.

| Optimizer | Pilot learning rate | Pilot test macro-F1, mean ± SD |
| --- | ---: | ---: |
| SGD | 0.01 | 0.804 ± 0.033 |
| Adagrad | 0.01 | 0.893 ± 0.026 |
| RMSprop | 0.001 | 0.888 ± 0.041 |
| Adam | 0.001 | 0.884 ± 0.026 |

The pilot motivated a fairer final comparison: tune each optimizer on the same grid, increase to five paired seeds, and use a predefined decision rule. The final model also removed dropout and class weighting. The archived pilot code applies random flips and rotations, while the final trainer does not. Consequently, pilot-to-final differences cannot be attributed to learning-rate tuning alone.

## Learning-rate sweep

Each cell is **validation macro-F1 from one run at seed 100**, transcribed from slide 11. Bold marks the selected learning rate for each optimizer.

| Learning rate | SGD | Adagrad | RMSprop | Adam |
| --- | ---: | ---: | ---: | ---: |
| 0.1 | **0.904** | 0.848 | 0.565 | 0.224 |
| 0.03 | 0.870 | **0.923** | 0.625 | 0.224 |
| 0.01 | 0.832 | 0.874 | 0.853 | 0.887 |
| 0.003 | 0.262 | 0.835 | 0.933 | **0.922** |
| 0.001 | 0.449 | 0.784 | **0.947** | 0.886 |
| 0.0003 | 0.240 | 0.573 | 0.899 | 0.920 |

Selected rates are **SGD 0.1, Adagrad 0.03, RMSprop 0.001, and Adam 0.003**, matching the final CSV. The presentation reports collapse to powdery-mildew-only predictions for Adam at 0.1 and 0.03. SGD's best rate lies at the grid boundary, so a wider grid could change its outcome.

## Small-CNN results

### Classification performance

Results use **five seeds per optimizer**, the same **538 test crops**, and a **CPU**. Mean ± SD uses the sample standard deviation (`ddof=1`). Accuracy values are proportions. Train and validation accuracy are means across seeds.

| Optimizer | Learning rate | Test macro-F1 | Test accuracy | Train accuracy | Validation accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| SGD | 0.1 | 0.858 ± 0.008 | 0.913 ± 0.006 | 0.912 | 0.936 |
| Adagrad | 0.03 | 0.883 ± 0.015 | 0.926 ± 0.011 | 0.950 | 0.939 |
| RMSprop | 0.001 | 0.898 ± 0.025 | 0.933 ± 0.017 | 0.958 | 0.948 |
| Adam | 0.003 | 0.907 ± 0.023 | 0.939 ± 0.013 | 0.974 | 0.953 |

![Small-CNN test performance and per-class F1](docs/figures/smallcnn/Figure_2_test_performance.png)

Adam has the highest observed mean test macro-F1, followed by RMSprop and Adagrad. Its advantage over SGD is **0.049 macro-F1**. The difference is largest for angular leaf spot, where mean F1 rises from **0.711 with SGD to 0.822 with Adam**. Powdery mildew F1 is high for all four optimizers (0.972–0.979).

### Training curves

![Small-CNN training and validation curves](docs/figures/smallcnn/Figure_1_training_curves.png)

Adam and RMSprop reach the target earlier than SGD. Training accuracy continues to improve while validation loss flattens or fluctuates, consistent with some overfitting. The plotted mean at later epochs contains only runs that are still training, so the contributing runs change after early stopping.

### Convergence and training time

All **20/20 runs** reached validation macro-F1 **0.85**. Epochs are mean ± SD; time to target and best epoch are means. Seconds per epoch is the **median of the five run-level values**.

| Optimizer | Epochs to 0.85 | Time to 0.85 (s) | Median s/epoch | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| SGD | 17.4 ± 2.1 | 292.28 | 17.48 | 27.0 |
| Adagrad | 10.4 ± 2.6 | 171.50 | 16.43 | 24.4 |
| RMSprop | 6.2 ± 2.2 | 111.53 | 15.75 | 23.8 |
| Adam | 6.0 ± 1.9 | 102.76 | 16.54 | 28.0 |

![Small-CNN convergence and time to target](docs/figures/smallcnn/Figure_3_convergence_and_time.png)

Adam and RMSprop need about six epochs, compared with 10.4 for Adagrad and 17.4 for SGD. Typical epoch times are similar, so epoch counts account for much of the time-to-target difference. The deck reports laptop pauses in SGD seeds 0 and 3 and Adagrad seed 3 after reaching the target; these make mean epoch times misleading. Timing includes training and validation work through the target epoch and is specific to this environment.

### Per-disease precision, recall, and F1

Values are means across five seeds, calculated directly from the final CSV.

| Optimizer | Disease | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| SGD | Angular leaf spot | 0.742 | 0.683 | 0.711 |
| SGD | Leaf spot | 0.869 | 0.913 | 0.890 |
| SGD | Powdery mildew leaf | 0.983 | 0.961 | 0.972 |
| Adagrad | Angular leaf spot | 0.777 | 0.762 | 0.768 |
| Adagrad | Leaf spot | 0.905 | 0.902 | 0.904 |
| Adagrad | Powdery mildew leaf | 0.973 | 0.978 | 0.976 |
| RMSprop | Angular leaf spot | 0.778 | 0.841 | 0.803 |
| RMSprop | Leaf spot | 0.932 | 0.898 | 0.914 |
| RMSprop | Powdery mildew leaf | 0.974 | 0.980 | 0.977 |
| Adam | Angular leaf spot | 0.788 | 0.866 | 0.822 |
| Adam | Leaf spot | 0.936 | 0.905 | 0.920 |
| Adam | Powdery mildew leaf | 0.978 | 0.980 | 0.979 |

Angular leaf spot shows the largest recall difference: **0.683 for SGD versus 0.866 for Adam**. This improvement accompanies some additional confusion of leaf spot with angular leaf spot.

![Small-CNN representative confusion matrices](docs/figures/smallcnn/Figure_4_confusion_matrices.png)

Each matrix represents the seed with the median test macro-F1 for that optimizer. Cells show counts and percentages within each true class. These are individual-run matrices, not pooled results across all seeds. One angular leaf spot test crop corresponds to about 1.7 percentage points of class recall.

### Generalization

![Small-CNN train, validation, and test comparison](docs/figures/smallcnn/Figure_S1_generalisation.png)

Validation and test macro-F1 have a pooled correlation of **r = 0.609** across the 20 runs. Validation scores generally exceed test scores. Training accuracy is accumulated during the selected epoch, whereas validation accuracy is measured after that epoch, which helps explain why SGD's reported training accuracy can be lower than validation accuracy.

### Decision

The recalculated Friedman test gives **p = 0.0136**. Applying the repository's paired tests and Holm correction leaves **Adam, RMSprop, and Adagrad** unresolved after the accuracy, convergence, and timing stages. Adam's epochs-to-target comparison with Adagrad has adjusted **p ≈ 0.06**.

**No single best optimizer is established by the predefined rule.** Adam leads the observed means, and RMSprop is close on performance and convergence. The full recalculated comparisons are in [significance.csv](docs/results/smallcnn/significance.csv), with the method recorded in [statistics.json](docs/results/smallcnn/statistics.json).

## ResNet-18 comparison

### Purpose and setup

The presentation asks whether a stronger pretrained model separates optimizer behavior. It uses ImageNet-pretrained ResNet-18 with only `layer4` and a new three-class classifier trained, **224 × 224** inputs, and ImageNet normalization. The same split and 538 test crops are used, with a 40-epoch maximum, an independent learning-rate sweep at seed 100, then five paired final seeds.

This is a second experimental condition. It changes pretraining, architecture depth, input resolution, and hardware together. The small CNN ran on CPU; ResNet-18 ran on GPU. Their training times do not form a controlled speed comparison.

### Model comparison

The small-CNN columns match the raw CSV. ResNet-18 values are rounded results reported in slides 18–21; its raw run CSV and training implementation are not included in this repository snapshot.

| Optimizer | Small-CNN macro-F1 | ResNet-18 macro-F1 | Small-CNN mean errors / 538 | ResNet-18 mean errors / 538 | ResNet-18 learning rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| SGD | 0.858 ± 0.008 | 0.986 ± 0.003 | 46.8 | 5.6 | 0.01 |
| Adagrad | 0.883 ± 0.015 | 0.991 ± 0.003 | 39.8 | 3.6 | 0.001 |
| RMSprop | 0.898 ± 0.025 | 0.991 ± 0.004 | 35.8 | 3.8 | 0.0001 |
| Adam | 0.907 ± 0.023 | 0.990 ± 0.003 | 32.8 | 4.4 | 0.0003 |

![Small CNN and pretrained ResNet-18 across optimizers](docs/figures/comparison/model_comparison.png)

The presentation reports that every ResNet-18 run exceeded every small-CNN run: the lowest ResNet-18 macro-F1 was approximately 0.983, versus a maximum of approximately 0.934 for the small CNN. Mean test errors fall by roughly **87–91%**, depending on optimizer. Angular leaf spot F1 improves from 0.71–0.82 to 0.98–0.99.

The spread between the highest and lowest optimizer means narrows from about **0.049 to 0.005**. This supports a benefit for the pretrained pipeline on this split; it does not isolate the contribution of architecture alone.

<details>
<summary>ResNet-18 test-performance figure and full model summary</summary>

![ResNet-18 performance across five seeds](docs/figures/resnet18/Figure_2_test_performance.png)

![Full comparison table from presentation slide 20](docs/figures/comparison/model_summary.png)

The repository's original [combined figure PDF](docs/figures/resnet18/All_figures.pdf) and its five PNG figures describe this ResNet-18 condition. They are stored separately from the small-CNN figures extracted from slides 12–16.

</details>

### Does ResNet-18 distinguish the optimizers?

Slide 27 reports no significant performance difference (Friedman **p = 0.21**, Adam versus RMSprop **p = 0.55**). All 20 runs reached its **0.80 validation macro-F1 target in the first epoch**, making that convergence threshold uninformative. Its roughly 7.8–8.1-second times to target measure one epoch, rather than a meaningful difference in learning speed.

The **0.80 ResNet-18 target differs from the 0.85 small-CNN target**. Neither time-to-target nor CPU/GPU epoch timings should be compared directly between models. ResNet-18 statistical claims are presentation-reported and could not be recalculated without its raw runs.

## Discussion and conclusion

The adaptive optimizers outperform plain SGD in the small CNN's observed means, particularly on the minority angular leaf spot class. Adam is a reasonable starting choice for the configuration tested, with RMSprop close behind, but the five-seed decision rule does not establish Adam as uniquely best.

The pretrained ResNet-18 pipeline produces a larger observed performance gain than switching optimizers within the small CNN. Its near-ceiling scores also make optimizer differences harder to resolve.

### Limitations and next steps

- Five seeds and one fixed dataset split provide limited evidence about robustness across runs or new growing conditions.
- Learning rates were selected with one sweep seed and a coarse grid. SGD's selected rate sits at the upper boundary.
- Only learning rate was tuned. Momentum, schedules, weight decay, and other optimizer settings were not systematically compared.
- There are only 58 angular leaf spot test crops. More independent examples and external validation are needed.
- The task classifies prepared leaf crops. Deployment would also need a method to locate leaves.
- The final small-CNN protocol has no augmentation. The pilot additionally differs in dropout, class weighting, and augmentation.
- The current sweep logs test metrics, despite selecting rates from validation only. Future experiments can defer test evaluation until final runs.
- The pretrained comparison changes several factors and hardware at once, and its low convergence target creates a ceiling effect.

The presentation proposes more seeds, SGD with momentum, data augmentation, and a separate AI Hub blight/wilt task as follow-up work.

## Repository structure

```text
StrawDisease_Optimizer_Comparison/
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── scripts/
│   ├── clean_datasets.py       # Dataset filtering, deduplication, cropping, splitting
│   ├── train_optimizers.py     # Main small-CNN smoke / sweep / final workflow
│   ├── analyze_optimizers.py   # Paired comparisons and result summaries
│   ├── make_figures.py         # Main figure generator
│   ├── compare_predictions.py # Compare existing predictions from two conditions
│   ├── pilot/train.py          # Original pilot implementation
│   └── legacy/make_figures_mdpi.py
└── docs/
    ├── PROVENANCE.md
    ├── presentation/           # Supplied research PowerPoint
    ├── results/smallcnn/       # Original CSV and recalculated statistical summaries
    └── figures/
        ├── smallcnn/          # Figures extracted from presentation slides 12–16
        ├── resnet18/          # Original repository figures and combined PDF
        └── comparison/        # Comparison images from slides 20–21
```

Large local folders such as `data/`, `strawberry_cls/`, `runs/`, `runs_resnet18/`, and newly generated `figures/` are ignored by Git. Curated research evidence under `docs/` is tracked.

## Setup and usage

### Install

Use Python 3.10 or newer. Dependency versions for the historical experiments were not supplied, so `requirements.txt` lists the required packages without claiming to recreate an exact original environment.

```bash
git clone https://github.com/isabelhendrick-sudo/StrawDisease_Optimizer_Comparison.git
cd StrawDisease_Optimizer_Comparison
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install compatible PyTorch and torchvision builds for the machine. The main trainer supports automatic device selection or explicit `--device cpu`, `cuda`, or `mps`.

### Prepare data

Obtain the YOLOv8 dataset export separately. Run from the repository root:

```bash
python scripts/clean_datasets.py "data/Strawberry Disease Detection Dataset.v4.zip" --out strawberry_cls
```

An extracted dataset folder can replace the ZIP path. It must contain `data.yaml` and split folders with `images/` and `labels/`. The script checks the expected class order. It refuses to overwrite a nonempty output directory.

The result contains `train/`, `validation/`, and `test/` class directories, plus `manifest.csv`, `duplicates.csv`, and `summary.txt`. Inspect the manifest and cleaning summary before training.

### Train and analyze the small CNN

```bash
python scripts/train_optimizers.py smoke --data strawberry_cls --out runs/smallcnn --device cpu
python scripts/train_optimizers.py sweep --data strawberry_cls --out runs/smallcnn --device cpu --img 128 --batch 32 --epochs 40 --patience 10 --sweep-seed 100 --lr-grid 0.1 0.03 0.01 0.003 0.001 0.0003 --target-f1 0.85
python scripts/train_optimizers.py final --data strawberry_cls --out runs/smallcnn --device cpu --img 128 --batch 32 --epochs 40 --patience 10 --seeds 0 1 2 3 4 --target-f1 0.85 --save-weights
python scripts/analyze_optimizers.py runs/smallcnn/final
python scripts/make_figures.py --runs runs/smallcnn/final --out figures/smallcnn
```

The CPU option matches the supplied small-CNN results. Other devices are available, but timings and numerical results may differ. Each phase writes aggregate `results.csv`, `history.csv`, and `predictions.csv`; the sweep writes `best_lr.json`, and `--save-weights` saves final-run checkpoints.

Completed runs are skipped when their `result.json` exists. Run names include optimizer, learning rate, and seed, so use a **new output directory** for changed data or settings. Final training loads selected rates from the sweep but does not recover other arguments automatically. All resized image splits are held in memory, which can require substantial RAM.

The repository's result snapshot contains only `results.csv`, not the original histories or predictions. The analysis and figure commands above require completed local runs with those additional files. Existing figures can be viewed directly in `docs/`.

### Pilot and model-comparison tools

```bash
# Original pilot configuration; results go beside the dataset under runs/<timestamp>_<tag>/
python scripts/pilot/train.py --data strawberry_cls --seeds 0 1 2 --epochs 40 --tag pilot

# Requires matching per-image predictions from both conditions and the same test data
python scripts/compare_predictions.py --base runs/smallcnn/final --new runs_resnet18/final --data strawberry_cls
```

The current `train_optimizers.py` implements the small CNN only. The ResNet-18 training code and raw runs must be added before that condition can be reproduced from this repository. `compare_predictions.py` relies on the same test-file order in both prediction files.

The original alternative plotting script is retained in `scripts/legacy/`; use `scripts/make_figures.py` for the main workflow.

## Git workflow and licensing

Use a short-lived branch for each change and keep code, research evidence, and generated experiments in their documented locations. See [CONTRIBUTING.md](CONTRIBUTING.md) for checks and publishing steps.

No code license file was present in the original repository. The dataset license reported by the presentation does not establish a license for this code or for third-party material in the deck.
