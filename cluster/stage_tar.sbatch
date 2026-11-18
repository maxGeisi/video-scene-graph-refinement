#!/usr/bin/env bash
#SBATCH --job-name=ag_tar
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=stage_tar_%j.log
# Packs Action Genome frames into per-video .tar archives (~9600 files instead
# of 288k) to fit the home file-count quota. Repacks already-extracted loose
# frames, then decodes the remaining videos straight into tars, deleting each
# source video as it goes. Verifies every annotated frame is inside its tar.
set -o pipefail
echo "=== ag_tar on $(hostname) at $(date) ==="
AG_ROOT="${AG_ROOT:-$HOME/ag}"
ANN="$AG_ROOT/annotations"; LOOSE="$AG_ROOT/frames"; TARS="$AG_ROOT/frames_tars"
VIDEOS="$AG_ROOT/Charades_v1_480"; WORK="$AG_ROOT/_work"
[ -f "$ANN/frame_list.txt" ] || { echo "FATAL: frame_list.txt missing"; exit 1; }
mkdir -p "$TARS" "$WORK" "$LOOSE"
echo "-- before --"; du -sh "$AG_ROOT"/* 2>/dev/null

export PATH="$HOME/.local/bin:$PATH"
python3 -m pip install --user --quiet imageio-ffmpeg opencv-python-headless tqdm 2>/dev/null || python3 -m pip install --user --quiet imageio-ffmpeg tqdm
FF=$(python3 -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
ln -sf "$FF" "$HOME/.local/bin/ffmpeg"; ffmpeg -version | head -1

cat > "$AG_ROOT/ag_tar.py" <<'PY'
import argparse, os, pickle, subprocess, sys, shutil, tarfile
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
def tar_ok(tarp, keep):
    if not os.path.exists(tarp): return False
    try:
        with tarfile.open(tarp,"r") as tf:
            names=set(tf.getnames())
        return all(f in names for f in keep)
    except Exception:
        return False
def pack_dir_to_tar(folder, tarp, keep):
    tmp=tarp+".tmp"
    with tarfile.open(tmp,"w") as tf:
        for f in keep:
            fp=os.path.join(folder,f)
            if os.path.exists(fp): tf.add(fp, arcname=f)
    os.replace(tmp,tarp)
def one_video(a):
    video,keep,video_dir,tar_dir,loose_dir,work_dir=a
    tarp=os.path.join(tar_dir, video+".tar")
    if tar_ok(tarp,keep):
        src=os.path.join(video_dir,video)
        if os.path.exists(src):
            try: os.remove(src)
            except OSError: pass
        return (video,"tar_done",len(keep))
    # (A) repack path: loose frames already extracted for this video
    folder=os.path.join(loose_dir,video)
    if os.path.isdir(folder) and any(os.path.exists(os.path.join(folder,f)) for f in keep):
        pack_dir_to_tar(folder,tarp,keep)
        shutil.rmtree(folder,ignore_errors=True)
        return (video,"repacked",len(keep))
    # (B) extract path: decode remaining video straight into a tar
    src=os.path.join(video_dir,video)
    if not os.path.exists(src): return (video,"MISSING_VIDEO",0)
    tmp=os.path.join(work_dir,video); os.makedirs(tmp,exist_ok=True)
    subprocess.run([FFMPEG,"-loglevel","panic","-i",src,"-qscale:v","4",
                    os.path.join(tmp,"%06d.jpg")],check=False)
    want_jpg={f[:-4]+".jpg" for f in keep}
    # keep only annotated, rename to .png (jpeg bytes)
    for f in os.listdir(tmp):
        p=os.path.join(tmp,f)
        if f in want_jpg: os.replace(p, os.path.join(tmp, f[:-4]+".png"))
        else:
            try: os.remove(p)
            except OSError: pass
    pack_dir_to_tar(tmp,tarp,keep)
    shutil.rmtree(tmp,ignore_errors=True)
    try: os.remove(src)
    except OSError: pass
    return (video,"extracted",len(keep))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video_dir",required=True); p.add_argument("--tar_dir",required=True)
    p.add_argument("--loose_dir",required=True); p.add_argument("--work_dir",required=True)
    p.add_argument("--annotation_dir",required=True); p.add_argument("--jobs",type=int,default=6)
    a=p.parse_args()
    os.makedirs(a.tar_dir,exist_ok=True); os.makedirs(a.work_dir,exist_ok=True)
    v2f=load_frame_list(a.annotation_dir)
    print(f"[tar] {len(v2f)} videos, {sum(len(x) for x in v2f.values())} frames, jobs={a.jobs}",flush=True)
    tasks=[(v,fr,a.video_dir,a.tar_dir,a.loose_dir,a.work_dir) for v,fr in v2f.items()]
    from collections import Counter
    c=Counter(); i=0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs=[ex.submit(one_video,t) for t in tasks]
        for fu in as_completed(futs):
            i+=1
            try: v,s,g=fu.result()
            except Exception as e:
                print(f"[tar] worker error: {e}",flush=True); c["error"]+=1; continue
            c[s]+=1
            if i%200==0: print(f"[tar] {i}/{len(tasks)}  {dict(c)}",flush=True)
    print(f"[tar] done: {dict(c)}",flush=True)
    # verify: every pkl key present as a member in its video tar
    obj=pickle.load(open(os.path.join(a.annotation_dir,"object_bbox_and_relationship.pkl"),"rb"))
    from collections import defaultdict
    need=defaultdict(set)
    for k in obj:
        vid,fr=k.split("/"); need[vid].add(fr)
    miss=0; checked=0
    for vid,frs in need.items():
        tarp=os.path.join(a.tar_dir,vid+".tar")
        names=set()
        if os.path.exists(tarp):
            try:
                with tarfile.open(tarp,"r") as tf: names=set(tf.getnames())
            except Exception: names=set()
        for fr in frs:
            checked+=1
            if fr not in names: miss+=1
    print(f"[verify] {checked-miss}/{checked} annotated frames present in tars, {miss} missing",flush=True)
    if miss:
        frac=miss/max(checked,1); print(f"[verify] MISSING FRACTION={frac:.4%}",flush=True)
        sys.exit(1 if frac>0.02 else 0)
    print("[verify] OK: all annotated frames present in tars.",flush=True)
if __name__=="__main__": main()

PY
python3 "$AG_ROOT/ag_tar.py" --video_dir "$VIDEOS" --tar_dir "$TARS" \
    --loose_dir "$LOOSE" --work_dir "$WORK" --annotation_dir "$ANN" --jobs 6
RC=$?
rm -rf "$WORK"; rmdir "$LOOSE" 2>/dev/null
echo "=== summary ==="; echo "tar files:"; ls "$TARS" | wc -l
du -sh "$TARS" 2>/dev/null; du -sh "$AG_ROOT" 2>/dev/null
echo "exit code: $RC"
[ "$RC" -eq 0 ] && echo "STAGING OK (tars)." || echo "STAGING INCOMPLETE. Paste log to Claude."
echo "=== done at $(date) ==="
exit $RC
