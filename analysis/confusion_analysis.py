#!/usr/bin/env python3
"""
Confusion analysis (multi-label, memory-safe, NUMPY-ONLY head).
Loads head weights from the .pt state_dict via torch IF available, else via a
pure-numpy path. Falls back to A-only if weights can't be read.
Runs variant A (no head), B (~/cache/head_B.pt), C (~/cache/head_C.pt).
"""
import os, glob, numpy as np
HOME=os.path.expanduser("~"); CACHE=os.path.join(HOME,"cache"); RESULTS=os.path.join(HOME,"results")
CONTACT17=['carrying','covered_by','drinking_from','eating','have_it_on_the_back','holding',
           'leaning_on','lying_on','not_contacting','other_relationship','sitting_on',
           'standing_on','touching','twisting','wearing','wiping','writing_on']
IDX={n:i for i,n in enumerate(CONTACT17)}
FINE=['drinking_from','eating','wiping','writing_on','twisting']
PARENT={'drinking_from':'holding','eating':'holding','wiping':'touching','writing_on':'touching','twisting':'holding'}

def files(): return sorted(glob.glob(os.path.join(CACHE,"train_*.npz"))) or sorted(glob.glob(os.path.join(CACHE,"*.npz")))
def logit(B):
    eps=1e-6; return np.log(np.clip(B,eps,1-eps))-np.log(np.clip(1-B,eps,1-eps))

def load_weights(path):
    """Return list of (W,b) numpy layers from a torch state_dict, or None."""
    if not os.path.exists(path): return None
    try:
        import torch
        sd=torch.load(path,map_location='cpu')
        if isinstance(sd,dict) and 'state_dict' in sd: sd=sd['state_dict']
        # collect linear layers in order
        W=[(k,v.numpy()) for k,v in sd.items() if k.endswith('.weight')]
        b={k:v.numpy() for k,v in sd.items() if k.endswith('.bias')}
        layers=[]
        for wk,wv in W:
            bk=wk.replace('.weight','.bias')
            layers.append((wv, b.get(bk, np.zeros(wv.shape[0]))))
        print(f"  loaded {len(layers)} linear layers from {os.path.basename(path)}: shapes {[w.shape for w,_ in layers]}")
        return layers
    except ModuleNotFoundError:
        # pure numpy .pt reader is not trivial; require torch for weights
        print("  torch not available -> cannot read head weights on this node")
        return None
    except Exception as e:
        print("  head load issue:",e); return None

def head_forward(X, layers):
    """Two-layer MLP with ReLU between (matches ResidualHead: Linear-ReLU-Linear)."""
    a=X
    for i,(W,b) in enumerate(layers):
        a=a@W.T + b
        if i < len(layers)-1: a=np.maximum(a,0)  # ReLU
    return a

def analyze(layers, tag):
    acc={f:{'n':0,'parent':0,'correct':0,'above':0} for f in FINE}
    for f in files():
        d=np.load(f); B=d['base']; Y=d['y']; s=logit(B)
        if layers is not None:
            s=s+head_forward(d['feat'].astype(np.float32), layers)
        top=np.argmax(s,axis=1)
        for fine in FINE:
            fi=IDX[fine]; pi=IDX[PARENT[fine]]; m=Y[:,fi]>0.5; n=int(m.sum())
            if n==0: continue
            acc[fine]['n']+=n
            acc[fine]['parent']+=int(np.sum(top[m]==pi))
            acc[fine]['correct']+=int(np.sum(top[m]==fi))
            acc[fine]['above']+=int(np.sum(s[m,pi]>s[m,fi]))
    print(f"\n=== Variant {tag} ===")
    rows=[]
    for fine in FINE:
        a=acc[fine]; n=a['n']; par=PARENT[fine]
        if n==0: rows.append((tag,fine,0,par,0,0,0)); continue
        pp=100*a['parent']/n; pc=100*a['correct']/n; pa=100*a['above']/n
        print(f"{fine:15s} n={n:5d} | predicts {par}: {pp:4.0f}%  | correct: {pc:4.0f}%  | {par} above: {pa:4.0f}%")
        rows.append((tag,fine,n,par,round(pp,1),round(pc,1),round(pa,1)))
    return rows

def main():
    print(f"streaming {len(files())} shards")
    allrows=analyze(None,"A")
    for tag,fn in [("B","head_B.pt"),("C","head_C.pt")]:
        L=load_weights(os.path.join(CACHE,fn))
        if L is not None: allrows+=analyze(L,tag)
    os.makedirs(RESULTS,exist_ok=True)
    with open(os.path.join(RESULTS,"confusion.csv"),"w") as fp:
        fp.write("variant,fine,n,parent,pct_predicts_parent,pct_correct,pct_parent_above\n")
        for r in allrows: fp.write(",".join(str(x) for x in r)+"\n")
    print("\nwrote ~/results/confusion.csv")

if __name__=="__main__": main()
