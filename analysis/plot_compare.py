"""Make the A/B/C comparison figures + table from variant_{A,B,C}.pkl.
Outputs (to ~/results):
  fig_motion_predicates.png : the 5 motion-defined predicates + 2 coarse parents, A vs B vs C
  fig_delta.png             : C-minus-B per predicate (does motion help?)
  comparison_table.csv      : full per-predicate mR for all contact predicates
"""
import os, pickle, argparse, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

MOTION=['drinking_from','eating','wiping','writing_on','twisting']
PARENTS=['holding','touching']
FOCUS=MOTION+PARENTS

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('-out',default=os.path.expanduser('~/results'))
    ap.add_argument('-k',type=int,default=20); ap.add_argument('-constraint',default='with')
    a=ap.parse_args()
    V={}
    for name in ['A','B','C']:
        p=os.path.join(a.out,f'variant_{name}.pkl')
        if os.path.exists(p): V[name]=pickle.load(open(p,'rb'))
    if not V: print('no variant pkls found in',a.out); return
    key=f'{a.constraint}_perpred@{a.k}'
    names=sorted({p for v in V.values() for p in v[key]})

    # full table
    import csv
    with open(os.path.join(a.out,'comparison_table.csv'),'w',newline='') as f:
        w=csv.writer(f); w.writerow(['predicate']+[f'{n}_mR@{a.k}' for n in V])
        for p in names: w.writerow([p]+[f'{V[n][key].get(p,0):.4f}' for n in V])
    print('wrote comparison_table.csv')

    # focus figure: motion predicates + parents
    labels=[p for p in FOCUS if p in names]
    x=np.arange(len(labels)); wdt=0.26
    plt.figure(figsize=(11,5))
    for i,n in enumerate(['A','B','C']):
        if n in V:
            vals=[V[n][key].get(p,0) for p in labels]
            plt.bar(x+(i-1)*wdt, vals, wdt, label={'A':'A: TEMPURA','B':'B: +head','C':'C: +head+motion'}[n])
    for i,p in enumerate(labels):
        if p in MOTION: plt.axvspan(i-0.4,i+0.4,color='orange',alpha=0.06)
    plt.xticks(x,labels,rotation=30,ha='right'); plt.ylabel(f'mean Recall@{a.k} ({a.constraint} constraint)')
    plt.title('Per-predicate mR: motion-defined predicates (shaded) vs coarse parents')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(a.out,'fig_motion_predicates.png'),dpi=150); plt.close()
    print('wrote fig_motion_predicates.png')

    # delta figure C-B
    if 'B' in V and 'C' in V:
        d=[V['C'][key].get(p,0)-V['B'][key].get(p,0) for p in labels]
        plt.figure(figsize=(11,4)); cols=['tab:green' if x>=0 else 'tab:red' for x in d]
        plt.bar(range(len(labels)),d,color=cols)
        plt.axhline(0,color='k',lw=0.8); plt.xticks(range(len(labels)),labels,rotation=30,ha='right')
        plt.ylabel(f'ΔmR@{a.k}  (C - B)'); plt.title('Effect of adding box-motion (C minus B)')
        plt.tight_layout(); plt.savefig(os.path.join(a.out,'fig_delta.png'),dpi=150); plt.close()
        print('wrote fig_delta.png')

if __name__=='__main__': main()
