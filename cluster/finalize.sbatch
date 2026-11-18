#!/usr/bin/env bash
#SBATCH --job-name=finalize
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=finalize_%j.log
# Trains head B and C on cached train features, evaluates A/B/C per-predicate on
# the test set (single model pass), and makes the comparison figures + table.
# Produces everything in $HOME/results.
set -o pipefail
say(){ echo "########## STAGE: $* ##########"; }
echo "=== finalize on $(hostname) at $(date) ==="
SCRATCH="/tmp/$USER-tempura"; CONDA_DIR="$SCRATCH/miniconda"; REPO="$SCRATCH/unbiasedSGG"
AG_ROOT="$HOME/ag"; export AG_TARS="$AG_ROOT/frames_tars"
CACHE="$HOME/cache"; OUT="$HOME/results"; mkdir -p "$OUT" "$SCRATCH"

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
  pip install --quiet "numpy==1.19.2" "scipy==1.7.3" "cython==0.29.33" h5py easydict "opencv-python==4.7.0.72" pandas "pytorch-metric-learning==2.1.1" "torchmetrics==0.11.4" tqdm gdown pyyaml dill matplotlib || { echo FATAL deps; exit 1; }
else conda activate TEMPURA; pip install --quiet matplotlib 2>/dev/null; fi
say "4 cython"; ( cd "$REPO/lib/draw_rectangles" && python setup.py build_ext --inplace ) || { echo FATAL cy1; exit 1; }
( cd "$REPO/lib/fpn/box_intersections_cpu" && python setup.py build_ext --inplace ) || { echo FATAL cy2; exit 1; }
say "5 shim + fixes"
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
say "7 copy scripts"
for f in train_head.py eval_head.py plot_compare.py embedding_distance.py; do
  cp "$HOME/$f" "$REPO/$f" || { echo "FATAL: put $f in HOME first"; exit 1; }
done
cd "$REPO"
say "8 TRAIN head B and C"
python -u train_head.py -cache "$CACHE" -epochs 15 2>&1 | tee "$HOME/train_head.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FATAL train_head"; exit 1; }
say "9 EVAL A/B/C per-predicate (single pass)"
python -u eval_head.py -data_path "$AG_ROOT/" -model_path "$MODEL" -cache "$CACHE" -out "$OUT" 2>&1 | tee "$HOME/eval_head.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FATAL eval_head"; exit 1; }
say "10 PLOTS + table"
python -u plot_compare.py -out "$OUT" -k 20 -constraint with 2>&1 | tee "$HOME/plots.log"
say "10b embedding-distance diagnostic (TEMPURA's own space)"
python -u embedding_distance.py -cache "$CACHE" -out "$OUT" 2>&1 | tee "$HOME/embedding_distance.log" || echo "WARN embedding_distance failed (non-fatal)"
say "11 summary"
ls -la "$OUT"
echo "FINALIZE OK -> $OUT"
echo "=== done at $(date) ==="
