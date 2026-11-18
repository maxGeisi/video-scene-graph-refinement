"""Evaluate variant A (baseline), B (head no motion), C (head + motion) on the
test set, producing PER-PREDICATE mean-Recall for each. For B/C we splice the
trained head's contact prediction into pred['contacting_distribution'] and run
TEMPURA's own evaluator. Writes per-predicate pickles + a text summary per variant.
"""
import os, sys, copy, argparse, numpy as np, torch, torch.nn as nn, pickle

def inv_sigmoid(p, eps=1e-6):
    p=torch.clamp(p, eps, 1-eps); return torch.log(p/(1-p))

class ResidualHead(nn.Module):
    def __init__(self, feat_dim=1936, n_contact=17, motion_dim=12, use_motion=True, hidden=512):
        super().__init__(); self.use_motion=use_motion
        if use_motion:
            self.motion=nn.Sequential(nn.BatchNorm1d(motion_dim),nn.Linear(motion_dim,128),nn.ReLU(),nn.Dropout(0.1))
            din=feat_dim+128
        else: din=feat_dim
        self.trunk=nn.Sequential(nn.Linear(din,hidden),nn.ReLU(),nn.Dropout(0.1))
        self.delta=nn.Linear(hidden,n_contact)
    def forward(self, base_logit, feat, motion=None):
        h=torch.cat([feat,self.motion(motion)],1) if self.use_motion else feat
        return base_logit + self.delta(self.trunk(h))

# motion builder identical to cache_features.py
def center_size(b):
    cx=(b[:,0]+b[:,2])/2; cy=(b[:,1]+b[:,3])/2
    w=(b[:,2]-b[:,0]).clamp(min=1); h=(b[:,3]-b[:,1]).clamp(min=1)
    return torch.stack([cx,cy,w,h],1)
def build_motion(entry):
    boxes=entry['boxes']; pair=entry['pair_idx']; im_idx=entry['im_idx']; labels=entry['pred_labels']
    cs=center_size(boxes[:,1:]); fno=boxes[:,0].long(); P=pair.shape[0]
    mo=torch.zeros(P,12,device=boxes.device)
    for k in range(P):
        s_i=pair[k,0].long(); o_i=pair[k,1].long(); f=int(im_idx[k].item())
        o_cls=labels[o_i]; s_cls=labels[s_i]
        pm=(fno<f)&(labels==o_cls)
        if pm.any():
            pf=fno[pm].max(); prev_o=((fno==pf)&(labels==o_cls)).nonzero(as_tuple=True)[0][0]
            gap=float(f-int(pf.item())) or 1.0
            fw=max(float(boxes[:,3].max()-boxes[:,1].min()),1.0); fh=max(float(boxes[:,4].max()-boxes[:,2].min()),1.0)
            norm=torch.tensor([fw,fh,fw,fh],device=boxes.device)
            do=((cs[o_i]-cs[prev_o])/norm)/gap; mo[k,0:4]=do; mo[k,4]=do[:2].norm()
            ps=(fno<f)&(labels==s_cls)
            if ps.any():
                psf=fno[ps].max(); prev_s=((fno==psf)&(labels==s_cls)).nonzero(as_tuple=True)[0][0]
                mo[k,5:9]=((cs[s_i]-cs[prev_s])/norm)/gap
            mo[k,9:11]=do[:2]-mo[k,5:7]; mo[k,11]=gap/30.0
    return mo

def per_pred_from_evaluator(ev, k):
    hit=ev.result_dict[ev.mode+'_recall_hit'][k]; cnt=ev.result_dict[ev.mode+'_recall_count'][k]
    out={}
    for idx in range(len(hit)):
        c=cnt[idx] if idx<len(cnt) else 0
        if c>0: out[ev.AG_all_predicates[idx]]=float(hit[idx])/float(c+1e-10)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('-data_path',required=True); ap.add_argument('-model_path',required=True)
    ap.add_argument('-cache',default=os.path.expanduser('~/cache')); ap.add_argument('-out',default=os.path.expanduser('~/results'))
    ap.add_argument('-head_dir',default=None, help='where head_B/C.pt live (default: -cache)')
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    sys.argv=['x','-mode','predcls','-datasize','large','-data_path',a.data_path,'-model_path',a.model_path,
              '-rel_mem_compute','joint','-rel_mem_weight_type','simple','-mem_fusion','late',
              '-mem_feat_selection','manual','-mem_feat_lambda','0.5','-rel_head','gmm','-obj_head','linear','-K','6']
    from lib.config import Config
    from dataloader.action_genome import AG, cuda_collate_fn
    from lib.object_detector import detector
    from lib.tempura import TEMPURA
    from lib.evaluation_recall import BasicSceneGraphEvaluator
    conf=Config(); gpu=torch.device('cuda:0')
    ds=AG(mode='test',datasize='large',data_path=conf.data_path,filter_nonperson_box_frame=True,filter_small_box=False)
    dl=torch.utils.data.DataLoader(ds,shuffle=False,num_workers=0,collate_fn=cuda_collate_fn)
    od=detector(train=False,object_classes=ds.object_classes,use_SUPPLY=True,mode=conf.mode).to(gpu); od.eval()
    model=TEMPURA(mode=conf.mode,attention_class_num=len(ds.attention_relationships),spatial_class_num=len(ds.spatial_relationships),
        contact_class_num=len(ds.contacting_relationships),obj_classes=ds.object_classes,enc_layer_num=conf.enc_layer,
        dec_layer_num=conf.dec_layer,obj_mem_compute=conf.obj_mem_compute,rel_mem_compute=conf.rel_mem_compute,
        take_obj_mem_feat=conf.take_obj_mem_feat,mem_fusion=conf.mem_fusion,selection=conf.mem_feat_selection,
        selection_lambda=conf.mem_feat_lambda,obj_head=conf.obj_head,rel_head=conf.rel_head,K=conf.K,tracking=conf.tracking).to(gpu); model.eval()
    ck=torch.load(conf.model_path,map_location=gpu); model.load_state_dict(ck['state_dict'],strict=True)
    contact_names=list(ds.contacting_relationships)

    # load heads B and C
    heads={}
    for name in ['B','C']:
        pth=os.path.join(a.head_dir or a.cache,f'head_{name}.pt')
        if os.path.exists(pth):
            st=torch.load(pth,map_location=gpu)
            h=ResidualHead(use_motion=st['use_motion']).to(gpu); h.load_state_dict(st['state_dict']); h.eval()
            heads[name]=h
    variants=['A']+list(heads.keys())

    # one evaluator per (variant, constraint)
    def mkev(c): return BasicSceneGraphEvaluator(mode='predcls',AG_object_classes=ds.object_classes,
        AG_all_predicates=ds.relationship_classes,AG_attention_predicates=ds.attention_relationships,
        AG_spatial_predicates=ds.spatial_relationships,AG_contacting_predicates=ds.contacting_relationships,
        output_dir=None,iou_threshold=0.5,constraint=c)
    EV={v:{c:mkev(c) for c in ['with','no']} for v in variants}

    # SINGLE model pass; evaluate all variants on the same forward
    with torch.no_grad():
        for b,data in enumerate(dl):
            im=copy.deepcopy(data[0].cuda(0)); ii=copy.deepcopy(data[1].cuda(0))
            gb=copy.deepcopy(data[2].cuda(0)); nb=copy.deepcopy(data[3].cuda(0))
            ga=ds.gt_annotations[data[4]]
            entry=od(im,ii,gb,nb,ga,im_all=None)
            pred=model(entry,phase='test',unc=False)
            base_contact=pred['contacting_distribution']
            feat=pred['rel_features'].float()
            mo=None
            for v in variants:
                if v=='A':
                    pred['contacting_distribution']=base_contact
                else:
                    h=heads[v]
                    if h.use_motion and mo is None: mo=build_motion(pred)
                    base_logit=inv_sigmoid(base_contact.float())
                    pred['contacting_distribution']=torch.sigmoid(h(base_logit, feat, mo if h.use_motion else None))
                pd=dict(pred)
                for c in ['with','no']:
                    EV[v][c].evaluate_scene_graph(ga, pd)
            if b%200==0: print(f'[eval] video {b}/{len(dl)}',flush=True)

    # dump per-predicate + mean for each variant
    for v in variants:
        summary={}
        for c in ['with','no']:
            ev=EV[v][c]
            for k in [10,20,50]:
                pp=per_pred_from_evaluator(ev,k)
                summary[f'{c}_mR@{k}']=float(np.mean([pp[p] for p in contact_names if p in pp])) if pp else 0.0
                summary[f'{c}_perpred@{k}']={p:pp.get(p,0.0) for p in contact_names}
        with open(os.path.join(a.out,f'variant_{v}.pkl'),'wb') as f: pickle.dump(summary,f)
        with open(os.path.join(a.out,f'variant_{v}.txt'),'w') as f:
            for c in ['with','no']:
                f.write(f'---------- {c} constraint ----------\n')
                for k in [10,20,50]:
                    f.write(f'mR@{k}: {summary[f"{c}_mR@{k}"]:.4f}\n')
                    for p in contact_names: f.write(f'    {p}: {summary[f"{c}_perpred@{k}"][p]:.4f}\n')
        print(f'[eval] wrote variant_{v}.txt / .pkl',flush=True)
    print('ALL VARIANTS DONE ->', a.out, flush=True)

if __name__=='__main__': main()
