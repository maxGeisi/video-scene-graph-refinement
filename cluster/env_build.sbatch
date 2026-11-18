#!/usr/bin/env bash
#SBATCH --job-name=env_build
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dortmund
#SBATCH --time=04:00:00
#SBATCH --output=env_build_%j.log
# Builds the TEMPURA environment on the node's LOCAL scratch (no home quota),
# clones the repo, compiles the two Cython bbox ops, installs a tar-reader shim
# so the dataloader reads frames from your per-video .tar archives. Pinned to
# dortmund so the scratch persists for the experiment job that follows.
set -o pipefail
echo "=== env_build on $(hostname) at $(date) ==="

SCRATCH="/scratch/$USER/tempura"
CONDA_DIR="$SCRATCH/miniconda"
REPO="$SCRATCH/unbiasedSGG"
AG_ROOT="$HOME/ag"                      # frame tars + annotations live in home (shared)
mkdir -p "$SCRATCH"
echo "scratch: $SCRATCH"; df -h "$SCRATCH" | tail -1

# ---------- 1. Miniconda on scratch ----------
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  echo "=== installing Miniconda on scratch ==="
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$SCRATCH/mc.sh" \
    || { echo "FATAL: miniconda download failed"; exit 1; }
  bash "$SCRATCH/mc.sh" -b -p "$CONDA_DIR" || { echo "FATAL: miniconda install failed"; exit 1; }
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda config --set always_yes yes >/dev/null 2>&1

# ---------- 2. repo ----------
if [ ! -d "$REPO" ]; then
  echo "=== cloning TEMPURA (unbiasedSGG) ==="
  git clone --depth 1 https://github.com/sayaknag/unbiasedSGG.git "$REPO" \
    || { echo "FATAL: git clone failed"; exit 1; }
fi

# ---------- 3. environment ----------
if ! conda env list | grep -q "TEMPURA"; then
  echo "=== creating TEMPURA env (python 3.7 + torch 1.10.1 cu111) ==="
  conda create -n TEMPURA python=3.7 -y || { echo "FATAL: conda create failed"; exit 1; }
  conda activate TEMPURA
  pip install --quiet "torch==1.10.1+cu111" "torchvision==0.11.2+cu111" \
      --extra-index-url https://download.pytorch.org/whl/cu111 \
      || { echo "FATAL: torch install failed"; exit 1; }
  pip install --quiet "numpy==1.19.2" "scipy==1.7.3" "cython==0.29.33" h5py easydict \
      "opencv-python==4.7.0.72" pandas "pytorch-metric-learning==2.1.1" \
      "torchmetrics==0.11.4" tqdm gdown \
      || { echo "FATAL: pip deps failed"; exit 1; }
else
  conda activate TEMPURA
fi
echo "python: $(python --version 2>&1)"

# ---------- 4. compile the two Cython bbox ops ----------
echo "=== compiling bbox ops ==="
( cd "$REPO/lib/draw_rectangles" && python setup.py build_ext --inplace ) \
    || { echo "FATAL: draw_rectangles compile failed"; exit 1; }
( cd "$REPO/lib/fpn/box_intersections_cpu" && python setup.py build_ext --inplace ) \
    || { echo "FATAL: box_intersections compile failed"; exit 1; }

# ---------- 5. tar-reader shim (read frames from per-video .tar) ----------
echo "=== installing tar-reader shim ==="
cat > "$REPO/dataloader/tar_frames.py" <<'PY'
import os, tarfile, threading
import numpy as np, cv2
_TARS = {}
_LOCK = threading.Lock()
_ROOT = os.environ.get("AG_TARS", "")   # set to <ag_root>/frames_tars
def imread(path, flag=cv2.IMREAD_UNCHANGED):
    # path ends with ".../frames/<video>.mp4/<frame>.png"; read from <video>.mp4.tar
    parts = path.replace("\\", "/").rstrip("/").split("/")
    video, frame = parts[-2], parts[-1]
    tarp = os.path.join(_ROOT, video + ".tar")
    with _LOCK:
        tf = _TARS.get(tarp)
        if tf is None:
            tf = tarfile.open(tarp, "r"); _TARS[tarp] = tf
        member = tf.extractfile(frame)
        data = member.read()
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, flag)
PY
# point the dataloader's imread at the shim
sed -i 's/^from cv2 import imread/from dataloader.tar_frames import imread/' "$REPO/dataloader/action_genome.py"
grep -n "tar_frames import imread" "$REPO/dataloader/action_genome.py" || echo "WARN: shim import not patched"

# ---------- 6. self-check ----------
echo "=== self-check ==="
python - <<PY
import torch, cv2, numpy
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only")
import sys; sys.path.insert(0, "$REPO")
import draw_rectangles  # compiled ext (draw_rectangles.pyx)
print("draw_rectangles import OK")
PY
RC=$?

echo "=== record paths for the experiment job ==="
cat > "$HOME/tempura_env.txt" <<TXT
CONDA_DIR=$CONDA_DIR
REPO=$REPO
SCRATCH=$SCRATCH
NODE=$(hostname)
AG_ROOT=$AG_ROOT
TXT
cat "$HOME/tempura_env.txt"

if [ "$RC" -eq 0 ]; then echo "ENV READY"; else echo "ENV BUILD INCOMPLETE. Paste log to Claude."; fi
echo "=== done at $(date) ==="
exit $RC
