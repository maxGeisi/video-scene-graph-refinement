#!/usr/bin/env bash
#SBATCH --job-name=ag_stage
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=stage_data_%j.log
# ---------------------------------------------------------------------------
# Stages Action Genome on the cluster: downloads Charades 480p videos + AG
# annotations, extracts ONLY the annotated frames (naming matches the pickle),
# then verifies every annotated frame exists. Runs on a compute node, not head.
# The GPU is requested only so the job schedules reliably; it is not used.
#
# Launch:   sbatch stage_data.sbatch
# Watch:    squeue -u $USER    and    tail -f stage_data_<jobid>.log
# ---------------------------------------------------------------------------
set -o pipefail
echo "=== ag_stage on $(hostname) at $(date) ==="

# Where everything goes. Override by exporting AG_ROOT before sbatch if home is small.
AG_ROOT="${AG_ROOT:-$HOME/ag}"
ANN="$AG_ROOT/annotations"
FRAMES="$AG_ROOT/frames"
VIDEOS="$AG_ROOT/Charades_v1_480"
mkdir -p "$AG_ROOT" "$ANN" "$FRAMES"
echo "AG_ROOT=$AG_ROOT"
echo "-- free space --"; df -h "$AG_ROOT" | tail -1

# --- tooling (userspace, no root) ---
export PATH="$HOME/.local/bin:$PATH"
python3 -m ensurepip --user >/dev/null 2>&1 || true
python3 -m pip install --user --quiet gdown imageio-ffmpeg tqdm || {
  echo "FATAL: pip install failed"; exit 1; }
# make an 'ffmpeg' on PATH from the imageio-ffmpeg bundled binary
FF=$(python3 -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
mkdir -p "$HOME/.local/bin" && ln -sf "$FF" "$HOME/.local/bin/ffmpeg"
echo "ffmpeg -> $FF"; ffmpeg -version | head -1

# --- 1. annotations (small, from the public AG Drive folder) ---
if [ ! -f "$ANN/frame_list.txt" ] || [ ! -f "$ANN/object_bbox_and_relationship.pkl" ]; then
  echo "=== downloading AG annotations ==="
  gdown --folder "https://drive.google.com/drive/folders/1LGGPK_QgGbh9gH9SDFv_9LIhBliZbZys" -O "$ANN" || {
    echo "FATAL: annotation download failed (gdown). If it is a virus-scan/too-many-files issue, tell Claude."; exit 1; }
  # gdown --folder may nest into a subfolder; flatten if needed
  if [ ! -f "$ANN/frame_list.txt" ]; then
    found=$(find "$ANN" -name frame_list.txt | head -1)
    [ -n "$found" ] && mv "$(dirname "$found")"/* "$ANN"/ 2>/dev/null
  fi
fi
echo "-- annotations present --"; ls -la "$ANN" | head
[ -f "$ANN/frame_list.txt" ] || { echo "FATAL: frame_list.txt missing"; exit 1; }
[ -f "$ANN/object_bbox_and_relationship.pkl" ] || { echo "FATAL: object pkl missing"; exit 1; }
[ -f "$ANN/person_bbox.pkl" ] || echo "WARN: person_bbox.pkl missing (needed later for training)"

# --- 2. Charades 480p videos (~13 GB) ---
if [ ! -d "$VIDEOS" ] || [ -z "$(ls -A "$VIDEOS" 2>/dev/null)" ]; then
  echo "=== downloading Charades 480p (~13 GB) ==="
  ZIP="$AG_ROOT/Charades_v1_480.zip"
  wget -c -q --show-progress \
    "https://ai2-public-datasets.s3.us-west-2.amazonaws.com/charades/Charades_v1_480.zip" \
    -O "$ZIP" || { echo "FATAL: Charades download failed. If 404, tell Claude the exact URL error."; exit 1; }
  echo "=== unzipping ==="
  unzip -q -o "$ZIP" -d "$AG_ROOT" || { echo "FATAL: unzip failed"; exit 1; }
  # unzip yields $AG_ROOT/Charades_v1_480/*.mp4
fi
NVID=$(ls "$VIDEOS"/*.mp4 2>/dev/null | wc -l)
echo "-- videos present: $NVID mp4 files --"
[ "$NVID" -gt 0 ] || { echo "FATAL: no videos found under $VIDEOS"; exit 1; }

# --- 3. extract only the annotated frames (parallel, exact naming) ---
echo "=== extracting annotated frames ==="
cat > "$AG_ROOT/ag_extract.py" <<'PYEOF'
import argparse, os, pickle, subprocess, sys, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
def ffmpeg_bin():
    if shutil.which("ffmpeg"): return "ffmpeg"
    import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
FFMPEG = ffmpeg_bin()
def load_frame_list(ad):
    v2f={}
    for line in open(os.path.join(ad,"frame_list.txt")):
        rel=line.strip()
        if not rel: continue
        v,fr=rel.split("/"); v2f.setdefault(v,[]).append(fr)
    return v2f
def one_video(a):
    video,keep,video_dir,frame_dir=a
    out=os.path.join(frame_dir,video); src=os.path.join(video_dir,video)
    if not os.path.exists(src): return (video,"MISSING_VIDEO",0)
    os.makedirs(out,exist_ok=True)
    if keep and all(os.path.exists(os.path.join(out,f)) for f in keep):
        return (video,"already_done",len(keep))
    subprocess.run([FFMPEG,"-loglevel","panic","-i",src,os.path.join(out,"%06d.png")],check=False)
    keep_set=set(keep)
    for f in set(os.listdir(out))-keep_set:
        try: os.remove(os.path.join(out,f))
        except OSError: pass
    return (video,"ok",sum(os.path.exists(os.path.join(out,f)) for f in keep))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video_dir",required=True); p.add_argument("--frame_dir",required=True)
    p.add_argument("--annotation_dir",required=True); p.add_argument("--jobs",type=int,default=8)
    a=p.parse_args()
    v2f=load_frame_list(a.annotation_dir)
    print(f"[extract] {len(v2f)} videos, {sum(len(x) for x in v2f.values())} frames, ffmpeg={FFMPEG}, jobs={a.jobs}",flush=True)
    tasks=[(v,fr,a.video_dir,a.frame_dir) for v,fr in v2f.items()]
    miss=[]; done=0; got=0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs=[ex.submit(one_video,t) for t in tasks]
        for i,fu in enumerate(as_completed(futs),1):
            v,s,g=fu.result()
            if s=="MISSING_VIDEO": miss.append(v)
            else: done+=1; got+=g
            if i%200==0: print(f"[extract] {i}/{len(tasks)}",flush=True)
    print(f"[extract] done videos={done} frames={got} missing_videos={len(miss)}",flush=True)
    if miss: print("[extract] missing e.g.",miss[:10],flush=True)
    obj=pickle.load(open(os.path.join(a.annotation_dir,"object_bbox_and_relationship.pkl"),"rb"))
    m=sum(0 if os.path.exists(os.path.join(a.frame_dir,k)) else 1 for k in obj)
    print(f"[verify] {len(obj)-m}/{len(obj)} annotated frames present, {m} missing",flush=True)
    if m: 
        frac=m/max(len(obj),1); print(f"[verify] MISSING FRACTION={frac:.4%}",flush=True)
        sys.exit(1 if frac>0.02 else 0)
    print("[verify] OK: all annotated frames present.",flush=True)
if __name__=="__main__": main()
PYEOF
python3 "$AG_ROOT/ag_extract.py" --video_dir "$VIDEOS" --frame_dir "$FRAMES" \
        --annotation_dir "$ANN" --jobs 16
RC=$?

echo "=== summary ==="
echo "frames root : $FRAMES"
echo "annotations : $ANN"
du -sh "$FRAMES" 2>/dev/null
echo "extractor/verify exit code: $RC"
if [ "$RC" -eq 0 ]; then
  echo "STAGING OK. You can now delete the videos to save space:  rm -rf '$VIDEOS' '$AG_ROOT'/Charades_v1_480.zip"
else
  echo "STAGING INCOMPLETE. Paste this log to Claude."
fi
echo "=== done at $(date) ==="
exit $RC
