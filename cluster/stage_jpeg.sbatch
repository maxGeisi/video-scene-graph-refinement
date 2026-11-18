#!/usr/bin/env bash
#SBATCH --job-name=ag_jpeg
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=stage_jpeg_%j.log
# Re-extracts Action Genome frames as JPEG (named .png so the dataloader reads
# them unchanged) so the full set fits in the home quota. Assumes the previous
# job already downloaded annotations and Charades videos into ~/ag.
set -o pipefail
echo "=== ag_jpeg on $(hostname) at $(date) ==="
AG_ROOT="${AG_ROOT:-$HOME/ag}"
ANN="$AG_ROOT/annotations"; FRAMES="$AG_ROOT/frames"; VIDEOS="$AG_ROOT/Charades_v1_480"

# sanity: inputs from the previous run must exist
[ -f "$ANN/frame_list.txt" ] || { echo "FATAL: $ANN/frame_list.txt missing"; exit 1; }
[ -f "$ANN/object_bbox_and_relationship.pkl" ] || { echo "FATAL: object pkl missing"; exit 1; }
[ -d "$VIDEOS" ] || { echo "FATAL: $VIDEOS missing"; exit 1; }

# free space: drop the zip and any partial PNG frames from the failed run
echo "-- before cleanup --"; df -h "$AG_ROOT" | tail -1
rm -f "$AG_ROOT/Charades_v1_480.zip"
rm -rf "$FRAMES"
mkdir -p "$FRAMES"
echo "-- after cleanup --"; du -sh "$AG_ROOT"/* 2>/dev/null

# tooling
export PATH="$HOME/.local/bin:$PATH"
python3 -m pip install --user --quiet imageio-ffmpeg tqdm || { echo "FATAL: pip failed"; exit 1; }
FF=$(python3 -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
ln -sf "$FF" "$HOME/.local/bin/ffmpeg"; ffmpeg -version | head -1

echo "=== extracting as JPEG ==="
cat > "$AG_ROOT/ag_extract_jpeg.py" <<'PY'
import argparse, os, pickle, subprocess, sys, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
def ffmpeg_bin():
    if shutil.which("ffmpeg"): return "ffmpeg"
    import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
FFMPEG=ffmpeg_bin()
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
    subprocess.run([FFMPEG,"-loglevel","panic","-i",src,"-qscale:v","3",
                    os.path.join(out,"%06d.jpg")],check=False)
    want_png=set(keep); want_jpg={f[:-4]+".jpg" for f in want_png}
    for f in os.listdir(out):
        if f.endswith(".jpg") and f not in want_jpg:
            try: os.remove(os.path.join(out,f))
            except OSError: pass
    got=0
    for pngname in want_png:
        jp=os.path.join(out,pngname[:-4]+".jpg"); pp=os.path.join(out,pngname)
        if os.path.exists(jp): os.replace(jp,pp); got+=1
        elif os.path.exists(pp): got+=1
    return (video,"ok",got)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video_dir",required=True); p.add_argument("--frame_dir",required=True)
    p.add_argument("--annotation_dir",required=True); p.add_argument("--jobs",type=int,default=8)
    a=p.parse_args()
    v2f=load_frame_list(a.annotation_dir)
    print(f"[jpeg] {len(v2f)} videos, {sum(len(x) for x in v2f.values())} frames, jobs={a.jobs}",flush=True)
    tasks=[(v,fr,a.video_dir,a.frame_dir) for v,fr in v2f.items()]
    miss=[];done=0;got=0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs=[ex.submit(one_video,t) for t in tasks]
        for i,fu in enumerate(as_completed(futs),1):
            v,s,g=fu.result()
            if s=="MISSING_VIDEO": miss.append(v)
            else: done+=1; got+=g
            if i%200==0: print(f"[jpeg] {i}/{len(tasks)}",flush=True)
    print(f"[jpeg] done videos={done} frames={got} missing_videos={len(miss)}",flush=True)
    obj=pickle.load(open(os.path.join(a.annotation_dir,"object_bbox_and_relationship.pkl"),"rb"))
    m=sum(0 if os.path.exists(os.path.join(a.frame_dir,k)) else 1 for k in obj)
    print(f"[verify] {len(obj)-m}/{len(obj)} present, {m} missing",flush=True)
    if m:
        frac=m/max(len(obj),1); print(f"[verify] MISSING FRACTION={frac:.4%}",flush=True)
        sys.exit(1 if frac>0.02 else 0)
    print("[verify] OK: all annotated frames present.",flush=True)
if __name__=="__main__": main()
PY
python3 "$AG_ROOT/ag_extract_jpeg.py" --video_dir "$VIDEOS" --frame_dir "$FRAMES" --annotation_dir "$ANN" --jobs 8
RC=$?
echo "=== summary ==="; du -sh "$FRAMES" 2>/dev/null; df -h "$AG_ROOT" | tail -1
echo "exit code: $RC"
if [ "$RC" -eq 0 ]; then
  echo "JPEG STAGING OK. Freeing videos:"; rm -rf "$VIDEOS"; du -sh "$AG_ROOT" 2>/dev/null
  echo "STAGING OK."
else
  echo "STAGING INCOMPLETE. Paste log to Claude."
fi
echo "=== done at $(date) ==="
exit $RC
