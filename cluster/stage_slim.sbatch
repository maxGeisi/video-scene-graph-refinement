#!/usr/bin/env bash
#SBATCH --job-name=ag_slim
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=stage_slim_%j.log
# Extracts AG frames as JPEG and DELETES each source video right after, so the
# 16GB of videos melts away as small frames appear -> fits a tiny home quota.
set -o pipefail
echo "=== ag_slim on $(hostname) at $(date) ==="
AG_ROOT="${AG_ROOT:-$HOME/ag}"
ANN="$AG_ROOT/annotations"; FRAMES="$AG_ROOT/frames"; VIDEOS="$AG_ROOT/Charades_v1_480"
[ -f "$ANN/frame_list.txt" ] || { echo "FATAL: frame_list.txt missing"; exit 1; }
[ -d "$VIDEOS" ] || { echo "FATAL: videos dir missing"; exit 1; }

# free the partial frames from the failed run so there is headroom to start
echo "-- before --"; du -sh "$AG_ROOT"/* 2>/dev/null
rm -rf "$FRAMES"; mkdir -p "$FRAMES"
echo "-- after clearing partial frames --"; du -sh "$AG_ROOT"/* 2>/dev/null

export PATH="$HOME/.local/bin:$PATH"
python3 -m pip install --user --quiet imageio-ffmpeg tqdm || { echo "FATAL: pip failed"; exit 1; }
FF=$(python3 -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
ln -sf "$FF" "$HOME/.local/bin/ffmpeg"; ffmpeg -version | head -1

echo "=== extracting JPEG + deleting videos as we go ==="
cat > "$AG_ROOT/ag_extract_slim.py" <<'PY'
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
    if keep and all(os.path.exists(os.path.join(out,f)) for f in keep):
        if os.path.exists(src):
            try: os.remove(src)
            except OSError: pass
        return (video,"already_done",len(keep))
    if not os.path.exists(src): return (video,"MISSING_VIDEO",0)
    os.makedirs(out,exist_ok=True)
    subprocess.run([FFMPEG,"-loglevel","panic","-i",src,"-qscale:v","4",
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
    try: os.remove(src)
    except OSError: pass
    return (video,"ok",got)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video_dir",required=True); p.add_argument("--frame_dir",required=True)
    p.add_argument("--annotation_dir",required=True); p.add_argument("--jobs",type=int,default=6)
    a=p.parse_args()
    v2f=load_frame_list(a.annotation_dir)
    print(f"[slim] {len(v2f)} videos, {sum(len(x) for x in v2f.values())} frames, jobs={a.jobs}",flush=True)
    tasks=[(v,fr,a.video_dir,a.frame_dir) for v,fr in v2f.items()]
    miss=[];done=0;got=0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs=[ex.submit(one_video,t) for t in tasks]
        for i,fu in enumerate(as_completed(futs),1):
            try: v,s,g=fu.result()
            except Exception as e:
                print(f"[slim] worker error: {e}",flush=True); continue
            if s=="MISSING_VIDEO": miss.append(v)
            else: done+=1; got+=g
            if i%200==0: print(f"[slim] {i}/{len(tasks)}",flush=True)
    print(f"[slim] done videos={done} frames={got} missing_videos={len(miss)}",flush=True)
    obj=pickle.load(open(os.path.join(a.annotation_dir,"object_bbox_and_relationship.pkl"),"rb"))
    m=sum(0 if os.path.exists(os.path.join(a.frame_dir,k)) else 1 for k in obj)
    print(f"[verify] {len(obj)-m}/{len(obj)} present, {m} missing",flush=True)
    if m:
        frac=m/max(len(obj),1); print(f"[verify] MISSING FRACTION={frac:.4%}",flush=True)
        sys.exit(1 if frac>0.02 else 0)
    print("[verify] OK: all annotated frames present.",flush=True)
if __name__=="__main__": main()
PY
python3 "$AG_ROOT/ag_extract_slim.py" --video_dir "$VIDEOS" --frame_dir "$FRAMES" --annotation_dir "$ANN" --jobs 6
RC=$?
echo "=== summary ==="; du -sh "$FRAMES" 2>/dev/null; du -sh "$VIDEOS" 2>/dev/null; du -sh "$AG_ROOT" 2>/dev/null
echo "exit code: $RC"
[ "$RC" -eq 0 ] && echo "STAGING OK." || echo "STAGING INCOMPLETE. Paste log to Claude."
echo "=== done at $(date) ==="
exit $RC
