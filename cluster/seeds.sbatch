#!/bin/bash
#SBATCH --job-name=seeds
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --nodelist=frankfurt
#SBATCH --time=04:00:00
#SBATCH --output=seeds_%j.log

# Repeats variants B and C over several seeds. Only the head is retrained;
# the backbone stays frozen and its cached outputs are reused, so this is far
# cheaper than the original pipeline.
#

set -e
SEEDS="0 1 2 3 4"
CACHE="$HOME/cache"
AG_ROOT="$HOME/data/ActionGenome"      # adjust if different
MODEL="$HOME/checkpoints/tempura.tar"  # adjust if different

SCRATCH="/tmp/$USER-tempura"; CONDA_DIR="$SCRATCH/miniconda"
mkdir -p "$SCRATCH"

echo "########## environment ##########"
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$SCRATCH/mc.sh"
  bash "$SCRATCH/mc.sh" -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda config --set always_yes yes >/dev/null 2>&1
if ! conda env list | grep -q TEMPURA; then
  conda create -n TEMPURA python=3.7 -y -c conda-forge --override-channels
  conda activate TEMPURA
  pip install --quiet "torch==1.10.1+cu111" "torchvision==0.11.2+cu111" \
      --extra-index-url https://download.pytorch.org/whl/cu111
  pip install --quiet "numpy==1.19.2" tqdm
else
  conda activate TEMPURA
fi

for S in $SEEDS; do
  OUT="$HOME/results_seed$S"
  mkdir -p "$OUT"
  echo "########## seed $S ##########"

  # trains heads B and C for this seed into $OUT, then evaluates A/B/C
  python -u "$HOME/train_head.py" -cache "$CACHE" -epochs 15 -seed "$S" -head_dir "$OUT"
  python -u "$HOME/eval_head.py" -data_path "$AG_ROOT/" -model_path "$MODEL" \
         -cache "$CACHE" -head_dir "$OUT" -out "$OUT"

  echo "seed $S done: $(ls $OUT)"
done

echo "########## aggregate ##########"
python3 "$HOME/aggregate_seeds.py" $HOME/results_seed*
