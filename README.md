# Refining rare predicates in video scene graphs

University project (Praktikum) at the CAMP chair, TU Munich, 2026. Full write-up in [`report/report_Geisenheyner.pdf`](report/report_Geisenheyner.pdf).

In Action Genome, a few contact predicates like `holding` and `touching` make up most of the labels. The rare ones (`drinking_from`, `eating`, `wiping`, `writing_on`, `twisting`) are mostly more specific versions of those, and scene graph models recover them poorly.

I kept a pretrained TEMPURA model frozen and trained a small head on top that adds a correction to its contact logits. The last layer starts at zero, so training begins exactly at the baseline. One variant also gets a 12-number description of how the object and person boxes move between frames.

Results on the PredCls protocol (mean recall at K=20):

| | all predicates | 5 rare ones |
|---|---|---|
| A: TEMPURA, frozen | 0.403 | 0.069 |
| B: + correction head | 0.499 | 0.273 |
| C: + correction head + motion | 0.497 | 0.249 |
| same head, replacing instead of correcting | 0.371 | 0.019 |

The gain comes from predicting a correction, not from the motion feature. The same head trained to replace TEMPURA's output ends up below the baseline.

## Code

```
src/        cache_features.py   run frozen TEMPURA once, save features, logits, motion, labels
            train_head.py       train the correction head on the cache
            eval_head.py        evaluate A/B/C with TEMPURA's own evaluator
analysis/   diagnostics and figures for the report
cluster/    SLURM scripts used on the CAMP cluster
patches/    the small changes TEMPURA needs to build on a current setup
```

TEMPURA ([Nag et al., CVPR 2023](https://github.com/sayaknag/unbiasedSGG)) is not included. Clone it separately and apply what's described in [`patches/README.md`](patches/README.md).

## Running it

On the cluster, in order:

```bash
sbatch cluster/env_build.sbatch     # environment (Python 3.7, torch 1.10.1+cu111)
sbatch cluster/stage_data.sbatch    # Action Genome (stage_tar.sbatch packs frames per video)
sbatch cluster/baseline.sbatch      # reproduce TEMPURA numbers
sbatch cluster/cache_job.sbatch     # cache backbone outputs, ~376k training pairs
sbatch cluster/finalize.sbatch      # train heads, evaluate, plots
```

Partition, node names and paths at the top of each script are specific to the CAMP cluster.

Once the cache exists, most analysis scripts run on a CPU in seconds and only need `numpy`, `matplotlib` and `gensim`:

```bash
python3 analysis/confusion_analysis.py
python3 analysis/text_embedding_analysis.py --out results/
python3 analysis/motion_diagnostics.py --ag_root $AG_ROOT --out results/
```
