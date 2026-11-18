"""Embedding-distance diagnostic in TEMPURA's OWN representation space.
For each contact predicate, compute the centroid of cached rel_features over all
pairs carrying that predicate. Then show that each motion-defined fine predicate
sits closest to its coarse parent -> the model cannot separate them without an
extra signal (motion). Uses ~/cache (feat + y). Outputs table + heatmap + finding.
"""
import os, glob, argparse, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

# AG contacting predicate order (17). We map by name via relationship_classes if available.
CONTACT=['carrying','covered_by','drinking_from','eating','have_it_on_the_back','holding',
         'leaning_on','lying_on','not_contacting','other_relationship','sitting_on',
         'standing_on','touching','twisting','wearing','wiping','writing_on']
MOTION=['drinking_from','eating','wiping','writing_on','twisting']
PARENTS=['holding','touching']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('-cache',default=os.path.expanduser('~/cache'))
    ap.add_argument('-out',default=os.path.expanduser('~/results'))
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    fs=sorted(glob.glob(os.path.join(a.cache,'train_*.npz')))
    assert fs,'no train shards'
    # accumulate centroid sums per predicate (streaming, low memory)
    D=None; S=None; C=None
    for f in fs:
        d=np.load(f); feat=d['feat'].astype('f'); y=d['y']
        if S is None: S=np.zeros((y.shape[1],feat.shape[1]),'f'); C=np.zeros(y.shape[1],'f')
        for c in range(y.shape[1]):
            m=y[:,c]>0
            if m.any(): S[c]+=feat[m].sum(0); C[c]+=m.sum()
    cent=np.zeros_like(S)
    for c in range(S.shape[0]):
        if C[c]>0: cent[c]=S[c]/C[c]
    # L2 normalize for cosine distance
    norm=cent/ (np.linalg.norm(cent,axis=1,keepdims=True)+1e-9)
    cos=norm@norm.T                      # cosine similarity matrix
    dist=1-cos                           # cosine distance
    names=CONTACT if S.shape[0]==len(CONTACT) else [f'c{i}' for i in range(S.shape[0])]
    idx={n:i for i,n in enumerate(names)}

    # finding: for each motion predicate, nearest predicate + distance to each parent
    lines=[]
    with open(os.path.join(a.out,'embedding_distance.csv'),'w') as fcsv:
        fcsv.write('fine_predicate,nearest_overall,dist_nearest,dist_holding,dist_touching\n')
        for fp in MOTION:
            if fp not in idx or C[idx[fp]]==0: continue
            i=idx[fp]
            order=np.argsort(dist[i])
            nearest=[names[j] for j in order if j!=i][0]
            dh=dist[i,idx['holding']] if 'holding' in idx else float('nan')
            dt=dist[i,idx['touching']] if 'touching' in idx else float('nan')
            fcsv.write(f'{fp},{nearest},{dist[i,idx[nearest]]:.4f},{dh:.4f},{dt:.4f}\n')
            lines.append(f'{fp:14s} nearest={nearest:12s} d(holding)={dh:.3f} d(touching)={dt:.3f}')
    print('=== embedding-distance finding (TEMPURA rel_features centroids) ===')
    for l in lines: print(l)

    # heatmap of the focus predicates
    focus=[p for p in (MOTION+PARENTS) if p in idx and C[idx[p]]>0]
    sub=np.array([[dist[idx[a2],idx[b]] for b in focus] for a2 in focus])
    plt.figure(figsize=(7,6)); plt.imshow(sub,cmap='viridis')
    plt.colorbar(label='cosine distance'); plt.xticks(range(len(focus)),focus,rotation=45,ha='right')
    plt.yticks(range(len(focus)),focus); plt.title('TEMPURA rel_features: distance between predicates')
    for i in range(len(focus)):
        for j in range(len(focus)): plt.text(j,i,f'{sub[i,j]:.2f}',ha='center',va='center',color='w',fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(a.out,'fig_embedding_distance.png'),dpi=150); plt.close()
    print('wrote embedding_distance.csv + fig_embedding_distance.png to',a.out)

if __name__=='__main__': main()
