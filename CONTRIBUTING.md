# Contributing

## Repository conventions

- Keep the main experiment scripts in `scripts/`. The original pilot and alternate plotter live in `scripts/pilot/` and `scripts/legacy/`.
- Put selected figures and small research results in `docs/`, separated by model. Include the source, model, split, seeds, and aggregation method when adding evidence.
- Keep raw datasets, cropped images, checkpoints, caches, and complete run directories local. `.gitignore` excludes the default locations.
- Resolve default paths from the repository root. Explicit relative command-line paths resolve from the current working directory.
- Run commands in the README from the repository root. Moving a script also requires updating its path handling and documentation.

## Checks

```bash
python -m compileall -q scripts
git diff --check
python scripts/clean_datasets.py --help
python scripts/train_optimizers.py --help
python scripts/make_figures.py --help
python scripts/compare_predictions.py --help
```

The analysis and legacy plotter use positional inputs and do not implement `--help`. Check them using complete result files. For changes to training behavior, run the two-epoch smoke phase on prepared data in a new output directory. Syntax checks alone do not validate model behavior.

For documentation changes, check every local Markdown link, model-specific figure placement, and table calculation. Keep presentation-reported results distinct from values recalculated from raw CSVs. Record unresolved discrepancies in `docs/PROVENANCE.md`.

## Git workflow

```bash
git switch main
git pull --ff-only
git switch -c docs/update-research
# Make and check a focused change.
git status --short
git diff --check
git diff
git add README.md docs/
git commit -m "docs: update research results"
git push -u origin docs/update-research
```

Open a pull request into `main` with a description of what changed and how it was verified. Use `docs/`, `fix/`, or `feat/` branch prefixes and small commits. Review the staged file list before committing to avoid including local datasets or environments.

## Experiment records

Save the Git revision, full command lines, package versions, hardware, dataset export identifier, cleaning settings, and random seeds with each experiment. Use a fresh run directory for changed configurations because the final trainer skips completed run names.

Do not compare CPU and GPU timings as a model-speed experiment. Keep convergence thresholds explicit, and do not mix the small-CNN 0.85 target with the presentation's ResNet-18 0.80 target. A nonsignificant test is not an equivalence test.
