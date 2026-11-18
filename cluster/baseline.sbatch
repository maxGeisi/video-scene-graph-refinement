#!/usr/bin/env bash
#SBATCH --job-name=baseline
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=baseline_%j.log
# Reproduces the TEMPURA PredCls baseline (variant A): builds the env on scratch,
# compiles the detector CUDA ops, downloads detector + TEMPURA checkpoint, and
# runs eval reading frames from your per-video .tar archives. Writes the
# per-predicate mean-Recall to ~/baseline_result.txt. Clear STAGE markers so a
# failure tells us exactly where.
set -o pipefail
say(){ echo "########## STAGE: $* ##########"; }
echo "=== baseline on $(hostname) at $(date) ==="

SCRATCH="/tmp/$USER-tempura"; CONDA_DIR="$SCRATCH/miniconda"; REPO="$SCRATCH/unbiasedSGG"
AG_ROOT="$HOME/ag"; export AG_TARS="$AG_ROOT/frames_tars"
mkdir -p "$SCRATCH"; df -h "$SCRATCH" | tail -1

say "1 miniconda"
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$SCRATCH/mc.sh" || { echo FATAL mc-dl; exit 1; }
  bash "$SCRATCH/mc.sh" -b -p "$CONDA_DIR" || { echo FATAL mc-install; exit 1; }
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"; conda config --set always_yes yes >/dev/null 2>&1

say "2 repo"
[ -d "$REPO" ] || git clone --depth 1 https://github.com/sayaknag/unbiasedSGG.git "$REPO" || { echo FATAL clone; exit 1; }

say "3 env (python3.7 + torch1.10.1cu111)"
if ! conda env list | grep -q TEMPURA; then
  conda create -n TEMPURA python=3.7 -y -c conda-forge --override-channels || { echo FATAL env-create; exit 1; }
  conda activate TEMPURA
  pip install --quiet "torch==1.10.1+cu111" "torchvision==0.11.2+cu111" --extra-index-url https://download.pytorch.org/whl/cu111 || { echo FATAL torch; exit 1; }
  pip install --quiet "numpy==1.19.2" "scipy==1.7.3" "cython==0.29.33" h5py easydict "opencv-python==4.7.0.72" pandas "pytorch-metric-learning==2.1.1" "torchmetrics==0.11.4" tqdm gdown pyyaml dill || { echo FATAL deps; exit 1; }
else
  conda activate TEMPURA
fi
python --version

say "4 cython cpu ops"
( cd "$REPO/lib/draw_rectangles" && python setup.py build_ext --inplace ) || { echo FATAL cython1; exit 1; }
( cd "$REPO/lib/fpn/box_intersections_cpu" && python setup.py build_ext --inplace ) || { echo FATAL cython2; exit 1; }

say "5 detector ops via torchvision (no CUDA compile needed)"
cat > "$REPO/fasterRCNN/lib/model/roi_layers/__init__.py" <<'ROIPY'
import torch
from torch import nn
import torchvision.ops as tvops
def nms(dets, scores, iou_threshold):
    return tvops.nms(dets, scores, iou_threshold)
def roi_align(input, rois, output_size, spatial_scale, sampling_ratio):
    if isinstance(output_size, int): output_size=(output_size,output_size)
    return tvops.roi_align(input, rois, output_size=output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio, aligned=False)
class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super().__init__(); self.output_size=output_size; self.spatial_scale=spatial_scale; self.sampling_ratio=sampling_ratio
    def forward(self, input, rois):
        return roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio)
def roi_pool(input, rois, output_size, spatial_scale):
    if isinstance(output_size, int): output_size=(output_size,output_size)
    return tvops.roi_pool(input, rois, output_size=output_size, spatial_scale=spatial_scale)
class ROIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super().__init__(); self.output_size=output_size; self.spatial_scale=spatial_scale
    def forward(self, input, rois):
        return roi_pool(input, rois, self.output_size, self.spatial_scale)
__all__=["nms","roi_align","ROIAlign","roi_pool","ROIPool"]
ROIPY
python -c "import torchvision, torchvision.ops; print('torchvision ops available', torchvision.__version__)" || { echo "FATAL torchvision missing"; exit 1; }
echo "detector ops shimmed via torchvision"

say "6 tar-reader shim"
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
            m=tf.extractfile(frame)
            data=m.read() if m is not None else None
        if data:
            img=cv2.imdecode(np.frombuffer(data,np.uint8),flag)
            if img is not None: return img
    except (KeyError, tarfile.TarError, OSError):
        pass
    return np.zeros((480,640,3), dtype=np.uint8)
PYX
sed -i 's/^from cv2 import imread/from dataloader.tar_frames import imread/' "$REPO/dataloader/action_genome.py"
grep -q "tar_frames import imread" "$REPO/dataloader/action_genome.py" || echo "WARN shim not patched"
# remove the leftover debug exit() in the transformer forward + silence its prints
sed -i '/^        exit()$/d' "$REPO/lib/transformer.py"
sed -i 's/^\(        \)print(\(rel_input\|local_output\|global_input\|global_masks\).*/\1pass/' "$REPO/lib/transformer.py"
grep -n "exit()" "$REPO/lib/transformer.py" && echo "WARN exit still present" || echo "transformer exit() removed"
# dataloader joins root/frames/<name>; make it exist (shim ignores contents)
mkdir -p "$AG_ROOT/frames"

say "7 weights"
mkdir -p "$REPO/fasterRCNN/models" "$SCRATCH/ckpt"
[ -f "$REPO/fasterRCNN/models/faster_rcnn_ag.pth" ] || \
  gdown "1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs" -O "$REPO/fasterRCNN/models/faster_rcnn_ag.pth" || { echo FATAL detector-weights; exit 1; }
MODEL="$SCRATCH/ckpt/predcls/best_Mrecall_model.tar"
if [ ! -f "$MODEL" ]; then
  mkdir -p "$SCRATCH/ckpt/predcls"
  gdown "1wyaRokpVt-ACWrrbIAxpEraJ-Jsd1qsK" -O "$MODEL" || { echo FATAL ckpt-dl; exit 1; }
fi
echo "detector: $(ls -la "$REPO/fasterRCNN/models/faster_rcnn_ag.pth" 2>/dev/null)"
echo "tempura checkpoint chosen: $MODEL"
[ -n "$MODEL" ] || { echo "FATAL: no predcls checkpoint found in folder; contents:"; ls -la "$SCRATCH/ckpt"; exit 1; }

say "8 baseline eval (variant A) -> writes ~/baseline_result.txt"
cd "$REPO"
python test.py -mode predcls -datasize large -data_path "$AG_ROOT/" -model_path "$MODEL" \
  -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late \
  -mem_feat_selection manual -mem_feat_lambda 0.5 -rel_head gmm -obj_head linear -K 6 \
  2>&1 | tee "$HOME/baseline_result.txt"
RC=${PIPESTATUS[0]}
echo "eval exit code: $RC"
[ "$RC" -eq 0 ] && echo "BASELINE OK -> ~/baseline_result.txt" || echo "BASELINE FAILED. Paste log to Claude."
echo "=== done at $(date) ==="
exit $RC
