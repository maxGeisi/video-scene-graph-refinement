# Motion-Conditioned Coarse-to-Fine Predicate Refinement for Dynamic Scene Graphs

University project (Master's Praktikum) at the Chair for Computer Aided Medical Procedures & Augmented Reality (CAMP), TU Munich, 2026.

**Author:** Maximilian Geisenheyner
**Supervisors:** Yousef Yeganeh, Azade Farshad
**Report:** [`report/report_Geisenheyner.pdf`](report/report_Geisenheyner.pdf)

---

## What this is

Rare contact predicates in Action Genome (`drinking_from`, `eating`, `wiping`,
`writing_on`, `twisting`) are fine-grained refinements of a small number of
frequent coarse predicates (`holding`, `touching`), and what separates a child
from its parent is object motion. This project adds a **zero-initialised residual
refinement head** on top of a **frozen TEMPURA backbone**, optionally conditioned
on a 12-dimensional bounding-box motion descriptor, and evaluates three variants
under the PredCls protocol.

| Variant | Description | macro mR@20 | fine-5 mR@20 |
|---------|-------------|-------------|--------------|
| A | frozen TEMPURA (baseline) | 0.403 | 0.069 |
| B | + residual head, no motion | **0.499** | **0.273** |
| C | + residual head + motion | 0.497 | 0.249 |
| — | *ablation:* replace head instead of refine | 0.371 | 0.019 |

The main finding: the improvement comes from the residual formulation, not
from the motion feature. The same head trained to replace the backbone
prediction instead of correcting it ends up below the baseline.

---

## Relation to TEMPURA

The backbone is [TEMPURA (Nag et al., CVPR 2023)](https://github.com/sayaknag/unbiasedSGG),
which is itself built on [STTran](https://github.com/yrcong/STTran). **That code is
not vendored here.** This repository contains only the code written for this
project, plus a description of the three small changes needed to make TEMPURA build
and run on a current cluster image.

To reproduce, clone TEMPURA separately and apply
[`patches/README.md`](patches/README.md).

---

## Layout

```
src/                      the method
  cache_features.py       runs the frozen backbone once, caches features/logits/motion/targets
  train_head.py           trains the residual refinement head on the cache
  eval_head.py            evaluates variants A/B/C through TEMPURA's own evaluator

analysis/                 diagnostics and figures (all independent of training)
  motion_diagnostics.py       coarse-to-fine co-occurrence + per-predicate motion signature
  embedding_distance.py       pairwise distances between mean relation features
  confusion_analysis.py       what the model predicts when a fine predicate is true
  text_embedding_analysis.py  scale-free visual-vs-text proximity comparison
  aggregate_seeds.py          multi-seed aggregation + paired t-test
  plot_compare.py             result figures
  inspect_y.py                small cache-format sanity check

cluster/                  SLURM job scripts (see "Pipeline" below)
patches/                  the TEMPURA modifications needed to build on a modern image
report/                   the final report
```

---

## Pipeline

The four stages are independent; each writes artefacts the next one reads.

| Stage | Script | Output |
|-------|--------|--------|
| 0. environment | `cluster/env_build.sbatch` | conda env in node-local scratch |
| 1. data staging | `cluster/stage_data.sbatch` (+ `stage_*.sbatch` variants) | Action Genome on cluster storage |
| 2. baseline | `cluster/baseline.sbatch` | reproduced TEMPURA numbers |
| 3. caching | `cluster/cache_job.sbatch` → `src/cache_features.py` | `~/cache/train_*.npz` |
| 4. train + eval | `cluster/finalize.sbatch` → `src/train_head.py`, `src/eval_head.py` | `~/results/variant_{A,B,C}.txt` |

The SLURM headers (`--partition`, `--nodelist`) and the data/checkpoint paths
at the top of each script are specific to the CAMP cluster; adjust them for
another environment.

Diagnostics afterwards:

```bash
sbatch cluster/confusion.sbatch          # prediction-level confusion (Fig. 10)
sbatch cluster/seeds.sbatch              # multi-seed repeat + paired test (seed 0 = reported run)
python3 analysis/text_embedding_analysis.py --out results/   # Fig. 9b, Tab. 5
```

### Cache format

Each shard is a compressed `npz` with four arrays:

| key | shape | contents |
|-----|-------|----------|
| `feat` | `N × 1936` | TEMPURA relation features |
| `base` | `N × 17` | frozen backbone contact probabilities |
| `motion` | `N × 12` | box-motion descriptor |
| `y` | `N × 17` | multi-hot contact targets |

`y` being **multi-hot** is what makes the co-occurrence analysis and the
confusion analysis possible without re-running the backbone. Action Genome
annotates several contact predicates per pair, e.g. a pair can be both
`holding` and `drinking_from`.

Total: 376,483 training pairs across 38 shards.

---

## Reproducing individual results

Most diagnostics run on CPU in seconds because they only stream the cache:

```bash
# Table 4 / Figure 10 — the confusion result
python3 analysis/confusion_analysis.py

# Figure 9b / Table 5 — visual vs text proximity (downloads GloVe on first run)
python3 analysis/text_embedding_analysis.py --out results/

# Figures 1 and 2 — dataset diagnostics
python3 analysis/motion_diagnostics.py --ag_root $AG_ROOT --out results/
```

`confusion_analysis.py` applies the trained head as an explicit two-layer matrix
product in NumPy rather than importing the training code, so it needs neither a
GPU nor the full environment.

---

## Requirements

The backbone requires the original pinned stack (`torch==1.10.1+cu111`,
`numpy==1.19.2`, Python 3.7); see `cluster/env_build.sbatch`, which builds it
from scratch in node-local scratch space.

The analysis scripts are far less demanding and run on any recent Python 3 with
`numpy`, `matplotlib`, and (for the text comparison) `gensim`.

---

## Not included

Deliberately excluded from version control: the Action Genome dataset, backbone
checkpoints, the feature cache (~several GB), and SLURM logs. See `.gitignore`.

---

## License

Code in this repository is released under the MIT License (see `LICENSE`).
TEMPURA, STTran and Action Genome are subject to their own licenses and terms
of use and are not redistributed here.

## Acknowledgements

Built on TEMPURA (Nag et al., CVPR 2023) and STTran (Cong et al., ICCV 2021),
using the Action Genome dataset (Ji et al., CVPR 2020). Full citations are in the
report bibliography.
