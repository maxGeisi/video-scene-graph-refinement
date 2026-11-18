#!/bin/bash
#SBATCH --job-name=confusion
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --nodelist=frankfurt
#SBATCH --time=00:40:00
#SBATCH --output=confusion_%j.log

set -e
say(){ echo "########## $* ##########"; }
echo "=== confusion start $(date) ==="

SCRATCH="/tmp/$USER-tempura"; CONDA_DIR="$SCRATCH/miniconda"
mkdir -p "$SCRATCH"

say "1 miniconda"
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$SCRATCH/mc.sh" || { echo FATAL mc; exit 1; }
  bash "$SCRATCH/mc.sh" -b -p "$CONDA_DIR" || { echo FATAL mc2; exit 1; }
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"; conda config --set always_yes yes >/dev/null 2>&1

say "2 env"
if ! conda env list | grep -q TEMPURA; then
  conda create -n TEMPURA python=3.7 -y -c conda-forge --override-channels || { echo FATAL env; exit 1; }
  conda activate TEMPURA
  pip install --quiet "torch==1.10.1+cu111" "torchvision==0.11.2+cu111" --extra-index-url https://download.pytorch.org/whl/cu111 || { echo FATAL torch; exit 1; }
  pip install --quiet "numpy==1.19.2" tqdm || true
else
  conda activate TEMPURA
fi

say "3 confusion analysis"
python3 -u "$HOME/confusion_analysis.py"

echo "=== confusion done $(date) ==="
