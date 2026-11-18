#!/usr/bin/env python3
"""
motion_diagnostics.py
=====================

Two training-free, model-free diagnostics for the motion-aware dynamic scene
graph project. Runs on ONLY the Action Genome annotation file you already have
(object_bbox_and_relationship.pkl). No GPU, no model, no cluster, no extra data.

It produces:

  1. COARSE-TO-FINE MAPPING (README section 7, diagnostic 2)
     Contacting relationships in Action Genome are multi-label: a single
     person-object pair can be labelled e.g. both "holding" and "drinking_from"
     in the same frame. So for every rare, motion-defined predicate we can count
     which coarse parent it co-occurs with, straight from the labels. This gives
     the coarse-to-fine mapping the two-step head needs, read from data instead
     of guessed, plus a talk-ready co-occurrence figure.

  2. BOX-MOTION SIGNATURE PER PREDICATE (README section 7, diagnostic 3)
     For each object instance we measure how far its box moves between annotated
     frames, normalised by the object's own size (resolution free, no frame
     dimensions needed) and by the frame gap (a speed, not a raw jump). We then
     compare the motion magnitude of motion-defined predicates (drinking_from,
     eating, ...) against static ones (sitting_on, other_relationship, ...).
     This turns the project's premise into a measured observation.

Usage
-----
    python3 motion_diagnostics.py \
        --pkl /path/to/object_bbox_and_relationship.pkl \
        --out ./motion_diag_out \
        --split train

Notes
-----
- --split: "train", "test", or "all" (default train, since the head trains on
  train; the motion signature is informative on any split).
- The person box lives in a separate file (person_bbox.pkl) and is not needed
  here; object box motion is exactly the signal for "cup moves to mouth".
- The script prints the schema it detects (bbox format, label type) so you can
  sanity check that the parse matched your file. If anything looks off, paste me
  the printed "SCHEMA" block and I will adjust the parser.
"""

import argparse
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Canonical Action Genome contacting classes (order used by STTran/TEMPURA).
# Used only as a fallback when labels are stored as integer indices rather than
# strings. If your file stores strings, these are ignored.
# ----------------------------------------------------------------------------
AG_CONTACTING = [
    "carrying", "covered_by", "drinking_from", "eating", "have_it_on_the_back",
    "holding", "leaning_on", "lying_on", "not_contacting", "other_relationship",
    "sitting_on", "standing_on", "touching", "twisting", "wearing", "wiping",
    "writing_on",
]

# The rare, motion-defined contacting predicates that are the whole point.
MOTION_DEFINED = ["drinking_from", "eating", "wiping", "writing_on", "twisting"]

# The coarse contact predicates these are refinements of.
COARSE_PARENTS = ["holding", "touching", "carrying"]

# Predicates we treat as "static" reference (object usually barely moves).
STATIC_REFERENCE = [
    "sitting_on", "standing_on", "lying_on", "leaning_on",
    "other_relationship", "not_contacting", "wearing", "covered_by",
]


# ----------------------------------------------------------------------------
# Schema helpers: the released AG pkl varies a little between mirrors, so detect
# rather than assume.
# ----------------------------------------------------------------------------
def _to_label(x, class_list):
    """Coerce a single relationship entry to a lowercase string label."""
    if isinstance(x, str):
        return x.strip().lower()
    # torch tensor / numpy scalar / python int
    if hasattr(x, "item"):
        try:
            x = x.item()
        except Exception:
            pass
    if isinstance(x, (int, np.integer)):
        if class_list is not None and 0 <= int(x) < len(class_list):
            return class_list[int(x)]
        return f"idx_{int(x)}"
    return str(x).strip().lower()


def _labels(entry, key, class_list):
    """Return list of string labels for a relationship field (may be missing)."""
    vals = entry.get(key, None)
    if vals is None:
        return []
    if isinstance(vals, (str, int, np.integer)) or hasattr(vals, "item"):
        vals = [vals]
    out = []
    for v in vals:
        lab = _to_label(v, class_list)
        if lab and lab not in ("none", "nan"):
            out.append(lab)
    return out


def _bbox_xywh_to_cxcywh(b, fmt):
    """Return (cx, cy, w, h) from a raw bbox in the detected format, or None."""
    if b is None:
        return None
    b = np.asarray(b, dtype=np.float64).ravel()
    if b.shape[0] < 4 or not np.all(np.isfinite(b[:4])):
        return None
    x0, y0, a, c = b[0], b[1], b[2], b[3]
    if fmt == "xyxy":
        w, h = a - x0, c - y0
        cx, cy = (x0 + a) / 2.0, (y0 + c) / 2.0
    else:  # "xywh"
        w, h = a, c
        cx, cy = x0 + a / 2.0, y0 + c / 2.0
    if w <= 0 or h <= 0:
        return None
    return np.array([cx, cy, w, h])


def detect_schema(data, class_list_hint):
    """Inspect a handful of entries to detect bbox format and label type."""
    n_check = 0
    xyxy_votes, xywh_votes = 0, 0
    label_is_str = None
    example = None
    for frame, objs in data.items():
        for o in objs:
            b = o.get("bbox", None)
            if b is None:
                continue
            arr = np.asarray(b, dtype=np.float64).ravel()
            if arr.shape[0] >= 4 and np.all(np.isfinite(arr[:4])):
                # In xyxy, arr[2]>arr[0] and arr[3]>arr[1]. In xywh, arr[2],arr[3]
                # are width/height (positive, and often smaller than arr[0]/arr[1]).
                if arr[2] > arr[0] and arr[3] > arr[1] and arr[2] > 1 and arr[3] > 1:
                    xyxy_votes += 1
                else:
                    xywh_votes += 1
                if example is None:
                    example = arr[:4].tolist()
            # label type
            cr = o.get("contacting_relationship", None)
            if cr:
                first = cr[0] if isinstance(cr, (list, tuple)) else cr
                if label_is_str is None:
                    label_is_str = isinstance(first, str)
            n_check += 1
            if n_check >= 4000:
                break
        if n_check >= 4000:
            break
    fmt = "xyxy" if xyxy_votes >= xywh_votes else "xywh"
    class_list = None if label_is_str else (class_list_hint or AG_CONTACTING)
    return fmt, bool(label_is_str), class_list, example


def frame_num(frame_key):
    """Extract integer frame index from a key like 'ABC.mp4/000089.png'."""
    m = re.search(r"(\d+)\.\w+$", frame_key)
    return int(m.group(1)) if m else None


def video_id(frame_key):
    return frame_key.split("/")[0]


def get_split(entry):
    md = entry.get("metadata", {}) or {}
    return md.get("set", None)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True,
                    help="path to object_bbox_and_relationship.pkl")
    ap.add_argument("--out", default="./motion_diag_out")
    ap.add_argument("--split", default="train",
                    choices=["train", "test", "all"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[load] reading {args.pkl}")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    print(f"[load] {len(data)} frames in file")

    fmt, label_is_str, class_list, example = detect_schema(data, AG_CONTACTING)
    print("\n================ SCHEMA (sanity check me) ================")
    print(f"  bbox format detected : {fmt}   (example raw bbox: {example})")
    print(f"  labels stored as     : {'strings' if label_is_str else 'indices -> mapped via canonical AG order'}")
    print(f"  split filter         : {args.split}")
    print("==========================================================\n")

    # ---- Pass 1: group object instances per (video, class) in frame order ----
    # AG annotates effectively one instance per object class per video, so we
    # match instances across frames by (video, class).
    per_instance = defaultdict(list)   # (video, cls) -> list of (fnum, cxcywh, contact_labels)
    cooc = defaultdict(lambda: defaultdict(int))  # fine -> parent -> count
    marg = defaultdict(int)            # predicate -> frames present
    kept_frames = 0

    for frame, objs in data.items():
        if args.split != "all":
            # split is per-object in metadata; use the first object's set tag
            s = None
            for o in objs:
                s = get_split(o)
                if s:
                    break
            if s is not None and s != args.split:
                continue
        fnum = frame_num(frame)
        vid = video_id(frame)
        if fnum is None:
            continue
        kept_frames += 1
        for o in objs:
            if not o.get("visible", True):
                continue
            cls = o.get("class", None)
            box = _bbox_xywh_to_cxcywh(o.get("bbox", None), fmt)
            contacts = _labels(o, "contacting_relationship", class_list)
            if contacts:
                # co-occurrence + marginals
                cset = set(contacts)
                for c in cset:
                    marg[c] += 1
                for fine in cset:
                    for other in cset:
                        if other != fine:
                            cooc[fine][other] += 1
            if cls is not None and box is not None:
                per_instance[(vid, cls)].append((fnum, box, contacts))

    print(f"[pass1] frames kept after split filter: {kept_frames}")
    print(f"[pass1] object instance tracks: {len(per_instance)}")

    # ---- Pass 2: box motion between consecutive annotated frames ----
    # normalised speed = center displacement / (object scale * frame gap)
    speed_by_pred = defaultdict(list)   # predicate -> list of normalised speeds
    size_change_by_pred = defaultdict(list)
    gaps = []

    for (vid, cls), seq in per_instance.items():
        seq.sort(key=lambda t: t[0])
        for i in range(1, len(seq)):
            f0, b0, _ = seq[i - 1]
            f1, b1, contacts1 = seq[i]
            gap = f1 - f0
            if gap <= 0:
                continue
            gaps.append(gap)
            cx0, cy0, w0, h0 = b0
            cx1, cy1, w1, h1 = b1
            disp = np.hypot(cx1 - cx0, cy1 - cy0)
            scale = np.sqrt(((w0 * h0) + (w1 * h1)) / 2.0)  # geometric-ish mean size
            if scale <= 0:
                continue
            norm_speed = disp / (scale * gap)
            size_change = (abs(w1 - w0) + abs(h1 - h0)) / (scale * gap)
            # attribute this motion to the contacting labels present at frame f1
            for c in set(contacts1):
                speed_by_pred[c].append(norm_speed)
                size_change_by_pred[c].append(size_change)

    if gaps:
        gaps = np.array(gaps)
        print(f"[pass2] frame gaps: median={np.median(gaps):.0f}, "
              f"mean={gaps.mean():.1f}, p90={np.percentile(gaps,90):.0f}")

    # ------------------------------------------------------------------
    # OUTPUT 1: coarse-to-fine mapping
    # ------------------------------------------------------------------
    mapping = {}
    for fine in MOTION_DEFINED:
        total = marg.get(fine, 0)
        if total == 0:
            mapping[fine] = {"present_frames": 0, "parents": []}
            continue
        parents = []
        for parent, cnt in sorted(cooc[fine].items(), key=lambda kv: -kv[1]):
            parents.append({
                "parent": parent,
                "cooccur_frames": cnt,
                "p_parent_given_fine": round(cnt / total, 4),
            })
        mapping[fine] = {"present_frames": total, "parents": parents}

    with open(os.path.join(args.out, "coarse_to_fine_mapping.json"), "w") as f:
        json.dump({
            "note": "p_parent_given_fine = P(parent present | fine present) on same pair/frame",
            "marginal_frames_per_predicate": dict(sorted(marg.items(), key=lambda kv: -kv[1])),
            "mapping": mapping,
        }, f, indent=2)

    print("\n===== COARSE-TO-FINE MAPPING (read from labels) =====")
    for fine in MOTION_DEFINED:
        info = mapping[fine]
        if info["present_frames"] == 0:
            print(f"  {fine:14s}: not present under this label spelling")
            continue
        top = info["parents"][0] if info["parents"] else None
        if top:
            print(f"  {fine:14s}: {info['present_frames']:6d} frames | "
                  f"top parent = {top['parent']} "
                  f"(P={top['p_parent_given_fine']:.2f})")
        else:
            print(f"  {fine:14s}: {info['present_frames']:6d} frames | no co-occurring contact label")

    # ------------------------------------------------------------------
    # OUTPUT 2: motion signature figures + table
    # ------------------------------------------------------------------
    def summ(pred):
        v = np.array(speed_by_pred.get(pred, []))
        if v.size == 0:
            return None
        return dict(n=int(v.size), median=float(np.median(v)),
                    mean=float(v.mean()), p90=float(np.percentile(v, 90)))

    signature = {p: summ(p) for p in AG_CONTACTING}
    with open(os.path.join(args.out, "motion_signature.json"), "w") as f:
        json.dump({k: v for k, v in signature.items() if v}, f, indent=2)

    # Figure A: motion-defined vs static, median normalised speed, sorted bar
    groups = [("motion-defined", MOTION_DEFINED, "#c0392b"),
              ("coarse parent", COARSE_PARENTS, "#e67e22"),
              ("static ref", STATIC_REFERENCE, "#2c7fb8")]
    names, meds, colors = [], [], []
    for _, preds, col in groups:
        for p in preds:
            s = signature.get(p)
            if s:
                names.append(p)
                meds.append(s["median"])
                colors.append(col)
    order = np.argsort(meds)
    names = [names[i] for i in order]
    meds = [meds[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(names))))
    ax.barh(range(len(names)), meds, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("median normalised box speed  (center displacement / object size / frame gap)")
    ax.set_title("Box-motion signature per contacting predicate")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in groups]
    ax.legend(handles, [g[0] for g in groups], loc="lower right", fontsize=9)
    fig.tight_layout()
    figA = os.path.join(args.out, "motion_signature_by_predicate.png")
    fig.savefig(figA, dpi=150)
    plt.close(fig)

    # Figure B: distribution (boxplot) for a focused subset
    focus = [p for p in (MOTION_DEFINED + COARSE_PARENTS + ["sitting_on", "other_relationship"])
             if speed_by_pred.get(p)]
    data_box = [np.clip(np.array(speed_by_pred[p]), 0, np.percentile(speed_by_pred[p], 99)) for p in focus]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data_box, showfliers=False)
    ax.set_xticks(range(1, len(focus) + 1))
    ax.set_xticklabels(focus)
    ax.set_ylabel("normalised box speed")
    ax.set_title("Motion distribution: motion-defined vs coarse vs static (outliers hidden)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    figB = os.path.join(args.out, "motion_distribution_focus.png")
    fig.savefig(figB, dpi=150)
    plt.close(fig)

    # Figure C: coarse-to-fine co-occurrence heatmap (fine rows vs contact cols)
    cols = COARSE_PARENTS + ["touching"]
    cols = list(dict.fromkeys(COARSE_PARENTS))  # unique, keep order
    rows = [p for p in MOTION_DEFINED if marg.get(p, 0) > 0]
    M = np.zeros((len(rows), len(cols)))
    for r, fine in enumerate(rows):
        tot = marg.get(fine, 1)
        for c, parent in enumerate(cols):
            M[r, c] = cooc[fine].get(parent, 0) / tot
    fig, ax = plt.subplots(figsize=(1.4 * len(cols) + 3, 0.7 * len(rows) + 2))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for r in range(len(rows)):
        for c in range(len(cols)):
            ax.text(c, r, f"{M[r,c]:.2f}", ha="center", va="center",
                    color="black" if M[r, c] < 0.6 else "white", fontsize=9)
    ax.set_title("P(coarse parent | fine predicate)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    figC = os.path.join(args.out, "coarse_to_fine_heatmap.png")
    fig.savefig(figC, dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # findings.md
    # ------------------------------------------------------------------
    lines = []
    lines.append("# Motion diagnostics (data-only, no model)\n")
    lines.append(f"- Source: `{os.path.basename(args.pkl)}`")
    lines.append(f"- Split: {args.split}")
    lines.append(f"- Frames used: {kept_frames}")
    lines.append(f"- Object instance tracks: {len(per_instance)}\n")
    lines.append("## Coarse-to-fine mapping (from multi-label co-occurrence)\n")
    lines.append("For each motion-defined fine predicate, the coarse contact predicate it")
    lines.append("most often co-occurs with on the same pair. This is the mapping the")
    lines.append("two-step head refines within.\n")
    lines.append("| fine predicate | frames | top coarse parent | P(parent \\| fine) |")
    lines.append("|---|---:|---|---:|")
    for fine in MOTION_DEFINED:
        info = mapping[fine]
        if info["present_frames"] and info["parents"]:
            top = info["parents"][0]
            lines.append(f"| {fine} | {info['present_frames']} | {top['parent']} | {top['p_parent_given_fine']:.2f} |")
        else:
            lines.append(f"| {fine} | {info['present_frames']} | (none) | - |")
    lines.append("\n## Box-motion signature (normalised speed)\n")
    lines.append("Higher = the object box moves more between annotated frames, relative to")
    lines.append("its own size and the frame gap.\n")
    lines.append("| predicate | group | n | median | mean | p90 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    grp = {}
    for g, preds, _ in groups:
        for p in preds:
            grp[p] = g
    for p in AG_CONTACTING:
        s = signature.get(p)
        if s:
            lines.append(f"| {p} | {grp.get(p,'other')} | {s['n']} | {s['median']:.3f} | {s['mean']:.3f} | {s['p90']:.3f} |")
    lines.append("\n## Figures\n")
    lines.append("- `motion_signature_by_predicate.png`")
    lines.append("- `motion_distribution_focus.png`")
    lines.append("- `coarse_to_fine_heatmap.png`")
    with open(os.path.join(args.out, "findings.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[done] wrote outputs to {args.out}/")
    for fn in ["findings.md", "coarse_to_fine_mapping.json", "motion_signature.json",
               "motion_signature_by_predicate.png", "motion_distribution_focus.png",
               "coarse_to_fine_heatmap.png"]:
        print(f"        - {fn}")


if __name__ == "__main__":
    main()
