"""Cache per-pair features from the FROZEN TEMPURA model for head training.
For each person-object pair in each frame we store:
  rel_feat  : 1936-dim pair vector (entry['rel_features'])
  motion    : box-motion feature (delta center/size of the OBJECT box vs its
              previous annotated frame, normalized) -> used only by variant C
  gt_contact: 17-dim multi-label contact target (from annotations)
Saved as compressed .npz shards to node-local scratch (no home quota touched).
Run once for train, once for test.
"""
import os, sys, copy, argparse, numpy as np, torch

def parse():
    p=argparse.ArgumentParser()
    p.add_argument('-split', choices=['train','test'], required=True)
    p.add_argument('-data_path', required=True)
    p.add_argument('-model_path', required=True)
    p.add_argument('-out', required=True)
    return p.parse_known_args()[0]

def center_size(b):  # [x1,y1,x2,y2] -> cx,cy,w,h
    cx=(b[:,0]+b[:,2])/2; cy=(b[:,1]+b[:,3])/2
    w=(b[:,2]-b[:,0]).clamp(min=1); h=(b[:,3]-b[:,1]).clamp(min=1)
    return torch.stack([cx,cy,w,h],1)

def build_motion(entry):
    """For each pair, motion of its OBJECT box relative to the same object's box
    in the previous annotated frame of this clip. 12-dim:
      [dcx,dcy,dw,dh, |d|, dcx_s,dcy_s,dw_s,dh_s (subject), rel_dcx,rel_dcy, gap]
    Falls back to zeros when no previous frame (first appearance)."""
    boxes=entry['boxes']              # [N,5] col0=frame idx
    pair=entry['pair_idx']            # [P,2] indices into boxes
    im_idx=entry['im_idx']            # [P] frame per pair
    labels=entry['pred_labels']       # [N] class per box
    cs=center_size(boxes[:,1:])       # [N,4] cx,cy,w,h  (pixel)
    fno=boxes[:,0].long()
    P=pair.shape[0]
    mo=torch.zeros(P,12, device=boxes.device)
    # index: for each (frame, class) remember the box row, to find previous frame
    for k in range(P):
        s_i=pair[k,0].long(); o_i=pair[k,1].long()
        f=int(im_idx[k].item())
        # previous frame index that contains the same object class
        o_cls=labels[o_i]; s_cls=labels[s_i]
        prev_mask=(fno<f)&(labels==o_cls)
        if prev_mask.any():
            pf=fno[prev_mask].max()
            prev_o=((fno==pf)&(labels==o_cls)).nonzero(as_tuple=True)[0][0]
            d=cs[o_i]-cs[prev_o]
            gap=float(f-int(pf.item())) or 1.0
            fw=max(float(boxes[:,3].max()-boxes[:,1].min()),1.0)
            fh=max(float(boxes[:,4].max()-boxes[:,2].min()),1.0)
            norm=torch.tensor([fw,fh,fw,fh],device=boxes.device)
            do=(d/norm)/gap
            mo[k,0:4]=do; mo[k,4]=do[:2].norm()
            # subject motion
            ps=(fno<f)&(labels==s_cls)
            if ps.any():
                psf=fno[ps].max()
                prev_s=((fno==psf)&(labels==s_cls)).nonzero(as_tuple=True)[0][0]
                ds=((cs[s_i]-cs[prev_s])/norm)/gap
                mo[k,5:9]=ds
            mo[k,9:11]=(do[:2]-mo[k,5:7])   # relative subject-object motion
            mo[k,11]=gap/30.0
    return mo

def main():
    a=parse()
    sys.argv=['x','-mode','predcls','-datasize','large','-data_path',a.data_path,
              '-model_path',a.model_path,'-rel_mem_compute','joint','-rel_mem_weight_type','simple',
              '-mem_fusion','late','-mem_feat_selection','manual','-mem_feat_lambda','0.5',
              '-rel_head','gmm','-obj_head','linear','-K','6']
    from lib.config import Config
    from dataloader.action_genome import AG, cuda_collate_fn
    from lib.object_detector import detector
    from lib.tempura import TEMPURA
    conf=Config(); gpu=torch.device('cuda:0')
    ds=AG(mode=a.split, datasize='large', data_path=conf.data_path,
          filter_nonperson_box_frame=True, filter_small_box=False)
    dl=torch.utils.data.DataLoader(ds, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)
    od=detector(train=False, object_classes=ds.object_classes, use_SUPPLY=True, mode=conf.mode).to(gpu); od.eval()
    model=TEMPURA(mode=conf.mode, attention_class_num=len(ds.attention_relationships),
        spatial_class_num=len(ds.spatial_relationships), contact_class_num=len(ds.contacting_relationships),
        obj_classes=ds.object_classes, enc_layer_num=conf.enc_layer, dec_layer_num=conf.dec_layer,
        obj_mem_compute=conf.obj_mem_compute, rel_mem_compute=conf.rel_mem_compute,
        take_obj_mem_feat=conf.take_obj_mem_feat, mem_fusion=conf.mem_fusion,
        selection=conf.mem_feat_selection, selection_lambda=conf.mem_feat_lambda,
        obj_head=conf.obj_head, rel_head=conf.rel_head, K=conf.K, tracking=conf.tracking).to(gpu); model.eval()
    ck=torch.load(conf.model_path, map_location=gpu); model.load_state_dict(ck['state_dict'], strict=True)

    os.makedirs(a.out, exist_ok=True)
    nC=len(ds.contacting_relationships)
    feats=[]; motions=[]; ys=[]; bases=[]; shard=0; kept=0
    with torch.no_grad():
        for idx,data in enumerate(dl):
            im=copy.deepcopy(data[0].cuda(0)); ii=copy.deepcopy(data[1].cuda(0))
            gb=copy.deepcopy(data[2].cuda(0)); nb=copy.deepcopy(data[3].cuda(0))
            ga=ds.gt_annotations[data[4]]
            entry=od(im,ii,gb,nb,ga,im_all=None)
            _=model(entry, phase='test', unc=False)
            rel=entry['rel_features'].detach().float().cpu().numpy()      # [P,1936]
            base=entry['contacting_distribution'].detach().float().cpu().numpy()  # TEMPURA contact probs [P,17]
            mo=build_motion(entry).detach().float().cpu().numpy()         # [P,12]
            # GT contact multi-labels per pair: EXACTLY as train.py builds them
            # contact_label[i, entry["contacting_gt"][i]] = 1   (lines 214-217)
            cg=entry["contacting_gt"]                                     # list length P of index-lists
            P=len(cg); Y=np.zeros((P,nC), np.float32)
            for i in range(P):
                idxs=cg[i]
                Y[i, idxs]=1.0
            rel=rel[:P]; mo=mo[:P]; base=base[:P]
            feats.append(rel); motions.append(mo); ys.append(Y); bases.append(base); kept+=P
            if len(feats)>=200:
                np.savez_compressed(os.path.join(a.out,f'{a.split}_{shard:04d}.npz'),
                    feat=np.concatenate(feats), motion=np.concatenate(motions), y=np.concatenate(ys), base=np.concatenate(bases))
                feats=[]; motions=[]; ys=[]; bases=[]; shard+=1
            if idx%100==0: print(f'[cache:{a.split}] video {idx} pairs_kept={kept}', flush=True)
    if feats:
        np.savez_compressed(os.path.join(a.out,f'{a.split}_{shard:04d}.npz'),
            feat=np.concatenate(feats), motion=np.concatenate(motions), y=np.concatenate(ys), base=np.concatenate(bases))
    print(f'[cache:{a.split}] DONE shards={shard+1} pairs={kept}', flush=True)

if __name__=='__main__': main()
