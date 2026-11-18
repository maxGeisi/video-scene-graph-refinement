# Changes to the TEMPURA codebase

The backbone is not vendored in this repository. Clone it separately:

```bash
git clone https://github.com/sayaknag/unbiasedSGG.git
```

Three small changes are needed to build and run it on a current cluster image. They are written as descriptions rather than a diff, because upstream keeps changing and a stale patch file is worse than an explicit note. The cluster scripts (`baseline.sbatch`, `cache_job.sbatch`, `finalize.sbatch`) apply all three automatically after cloning.

## 1. Detector ops via torchvision

**Problem.** TEMPURA inherits hand-written CUDA extensions for NMS, ROIAlign and ROIPool from the Faster R-CNN lineage. They no longer compile against current CUDA toolkits, which blocks even the baseline.

**Fix.** Overwrite `fasterRCNN/lib/model/roi_layers/__init__.py` so that it exposes the same names (`nms`, `roi_align`, `ROIAlign`, `roi_pool`, `ROIPool`) backed by `torchvision.ops`:

```python
import torch
from torch import nn
import torchvision.ops as tvops


def nms(dets, scores, iou_threshold):
    return tvops.nms(dets, scores, iou_threshold)


def roi_align(input, rois, output_size, spatial_scale, sampling_ratio):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    return tvops.roi_align(input, rois, output_size=output_size,
                           spatial_scale=spatial_scale,
                           sampling_ratio=sampling_ratio,
                           aligned=False)   # matches the legacy op


class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return roi_align(input, rois, self.output_size,
                         self.spatial_scale, self.sampling_ratio)


def roi_pool(input, rois, output_size, spatial_scale):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    return tvops.roi_pool(input, rois, output_size=output_size,
                          spatial_scale=spatial_scale)


class ROIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return roi_pool(input, rois, self.output_size, self.spatial_scale)


__all__ = ["nms", "roi_align", "ROIAlign", "roi_pool", "ROIPool"]
```

`aligned=False` keeps the half-pixel convention of the original op, so the pretrained detector weights still fit. The Cython extensions under `lib/draw_rectangles` and `lib/fpn/box_intersections_cpu` build normally and are compiled by the cluster scripts.

## 2. Reading frames from per-video tar archives

**Problem.** Action Genome has about 288k annotated frames. As loose files they exceed the file-count quota of a cluster home directory.

**Fix.** Frames are staged as one tar archive per video, about 9,600 files in total (`cluster/stage_tar.sbatch`, which also checks that every annotated frame ended up in its archive). A small module `dataloader/tar_frames.py` (written by the cluster scripts) provides an `imread(path, flag)` that reads the frame from `$AG_TARS/<video>.tar`, and the import in `dataloader/action_genome.py` is redirected to it:

```bash
sed -i 's/^from cv2 import imread/from dataloader.tar_frames import imread/' dataloader/action_genome.py
```

If a frame is missing from its archive, this `imread` returns a black 640×480 image instead of raising, so run the check in `stage_tar.sbatch` before caching features.

## 3. Stray `exit()` in `lib/transformer.py`

Upstream `lib/transformer.py` contains a leftover debug `exit()` (with two `print` calls before it) inside the transformer's forward pass. It is removed with:

```bash
sed -i '/^        exit()$/d' lib/transformer.py
```

## No change needed for feature export

`src/cache_features.py` does not modify TEMPURA. After the standard forward pass, TEMPURA already stores the 1936-d relation features in `entry["rel_features"]` and the contact probabilities in `entry["contacting_distribution"]`; the script reads both from there.

## Verification

After these changes, `cluster/baseline.sbatch` reproduces the PredCls numbers. The report uses this reproduction, not the numbers quoted in the paper, as the reference for all comparisons.
