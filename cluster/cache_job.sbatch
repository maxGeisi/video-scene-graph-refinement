#!/usr/bin/env bash
#SBATCH --job-name=cache
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=cache_%j.log
# Caches per-pair features (1936 rel vector + 12 motion + 17 contact GT) from
# frozen TEMPURA to $HOME/cache (sharded .npz, survives job end).
# Usage: sbatch cache_job.sbatch <split>   where <split> = test | train
set -o pipefail
SPLIT="${1:-test}"
say(){ echo "########## STAGE: $* ##########"; }
echo "=== cache[$SPLIT] on $(hostname) at $(date) ==="
SCRATCH="/tmp/$USER-tempura"; CONDA_DIR="$SCRATCH/miniconda"; REPO="$SCRATCH/unbiasedSGG"
AG_ROOT="$HOME/ag"; export AG_TARS="$AG_ROOT/frames_tars"
OUT="$HOME/cache"; mkdir -p "$OUT" "$SCRATCH"

say "1 miniconda"
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$SCRATCH/mc.sh" || { echo FATAL mc; exit 1; }
  bash "$SCRATCH/mc.sh" -b -p "$CONDA_DIR" || { echo FATAL mc2; exit 1; }
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"; conda config --set always_yes yes >/dev/null 2>&1
say "2 repo"; [ -d "$REPO" ] || git clone --depth 1 https://github.com/sayaknag/unbiasedSGG.git "$REPO" || { echo FATAL clone; exit 1; }
say "3 env"
if ! conda env list | grep -q TEMPURA; then
  conda create -n TEMPURA python=3.7 -y -c conda-forge --override-channels || { echo FATAL env; exit 1; }
  conda activate TEMPURA
  pip install --quiet "torch==1.10.1+cu111" "torchvision==0.11.2+cu111" --extra-index-url https://download.pytorch.org/whl/cu111 || { echo FATAL torch; exit 1; }
  pip install --quiet "numpy==1.19.2" "scipy==1.7.3" "cython==0.29.33" h5py easydict "opencv-python==4.7.0.72" pandas "pytorch-metric-learning==2.1.1" "torchmetrics==0.11.4" tqdm gdown pyyaml dill || { echo FATAL deps; exit 1; }
else conda activate TEMPURA; fi
say "4 cython"; ( cd "$REPO/lib/draw_rectangles" && python setup.py build_ext --inplace ) || { echo FATAL cy1; exit 1; }
( cd "$REPO/lib/fpn/box_intersections_cpu" && python setup.py build_ext --inplace ) || { echo FATAL cy2; exit 1; }
say "5 torchvision shim + fixes"
cat > "$REPO/fasterRCNN/lib/model/roi_layers/__init__.py" <<'ROIPY'
import torch
from torch import nn
import torchvision.ops as tvops
def nms(dets, scores, iou_threshold): return tvops.nms(dets, scores, iou_threshold)
def roi_align(input, rois, output_size, spatial_scale, sampling_ratio):
    if isinstance(output_size,int): output_size=(output_size,output_size)
    return tvops.roi_align(input, rois, output_size=output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio, aligned=False)
class ROIAlign(nn.Module):
    def __init__(self,o,s,r): super().__init__(); self.output_size=o; self.spatial_scale=s; self.sampling_ratio=r
    def forward(self,i,r): return roi_align(i,r,self.output_size,self.spatial_scale,self.sampling_ratio)
def roi_pool(input, rois, output_size, spatial_scale):
    if isinstance(output_size,int): output_size=(output_size,output_size)
    return tvops.roi_pool(input, rois, output_size=output_size, spatial_scale=spatial_scale)
class ROIPool(nn.Module):
    def __init__(self,o,s): super().__init__(); self.output_size=o; self.spatial_scale=s
    def forward(self,i,r): return roi_pool(i,r,self.output_size,self.spatial_scale)
__all__=["nms","roi_align","ROIAlign","roi_pool","ROIPool"]
ROIPY
cat > "$REPO/dataloader/tar_frames.py" <<'PYX'
import os, tarfile, threading, numpy as np, cv2
_T={}; _L=threading.Lock(); _R=os.environ.get("AG_TARS","")
def imread(path, flag=cv2.IMREAD_UNCHANGED):
    p=path.replace("\\","/").rstrip("/").split("/"); video,frame=p[-2],p[-1]
    tp=os.path.join(_R,video+".tar")
    try:
        with _L:
            tf=_T.get(tp)
            if tf is None: tf=tarfile.open(tp,"r"); _T[tp]=tf
            m=tf.extractfile(frame); data=m.read() if m is not None else None
        if data:
            img=cv2.imdecode(np.frombuffer(data,np.uint8),flag)
            if img is not None: return img
    except (KeyError, tarfile.TarError, OSError): pass
    return np.zeros((480,640,3), dtype=np.uint8)
PYX
sed -i 's/^from cv2 import imread/from dataloader.tar_frames import imread/' "$REPO/dataloader/action_genome.py"
sed -i '/^        exit()$/d' "$REPO/lib/transformer.py"
mkdir -p "$AG_ROOT/frames"
say "6 detector weights + checkpoint"
mkdir -p "$REPO/fasterRCNN/models" "$SCRATCH/ckpt/predcls"
[ -f "$REPO/fasterRCNN/models/faster_rcnn_ag.pth" ] || gdown "1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs" -O "$REPO/fasterRCNN/models/faster_rcnn_ag.pth" || { echo FATAL det; exit 1; }
MODEL="$SCRATCH/ckpt/predcls/best_Mrecall_model.tar"
[ -f "$MODEL" ] || gdown "1wyaRokpVt-ACWrrbIAxpEraJ-Jsd1qsK" -O "$MODEL" || { echo FATAL ckpt; exit 1; }
say "7 run cache[$SPLIT]"
cp "$HOME/cache_features.py" "$REPO/cache_features.py" || { echo "FATAL: put cache_features.py in HOME first"; exit 1; }
cd "$REPO"
python -u cache_features.py -split "$SPLIT" -data_path "$AG_ROOT/" -model_path "$MODEL" -out "$OUT" 2>&1 | tee "$HOME/cache_${SPLIT}.log"
RC=${PIPESTATUS[0]}
say "8 summary"
ls -la "$OUT"/${SPLIT}_*.npz 2>/dev/null | wc -l
du -sh "$OUT" 2>/dev/null
echo "cache exit code: $RC"
[ "$RC" -eq 0 ] && echo "CACHE[$SPLIT] OK" || echo "CACHE[$SPLIT] FAILED. Paste log to Claude."
echo "=== done at $(date) ==="
exit $RC
