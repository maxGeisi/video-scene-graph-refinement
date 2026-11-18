"""Train the coarse-to-fine RESIDUAL contact head on cached features.
final_logit = logit(TEMPURA_contact) + refinement(features [+motion])
The refinement's last layer is ZERO-INIT, so at the start B/C == A exactly and
training only has to learn a correction. Class-balanced loss lifts rare predicates.
Variant B: refinement without motion.  Variant C: refinement with motion.
"""
import os, glob, argparse, numpy as np, torch, torch.nn as nn

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
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)   # start = identity (== A)
    def forward(self, base_logit, feat, motion=None):
        h=torch.cat([feat,self.motion(motion)],1) if self.use_motion else feat
        return base_logit + self.delta(self.trunk(h))

def load_split(cache_dir, split):
    fs=sorted(glob.glob(os.path.join(cache_dir,f'{split}_*.npz')))
    assert fs, f'no {split} shards in {cache_dir}'
    F=[];M=[];Y=[];B=[]
    for f in fs:
        d=np.load(f); F.append(d['feat']); M.append(d['motion']); Y.append(d['y']); B.append(d['base'])
    return np.concatenate(F),np.concatenate(M),np.concatenate(Y),np.concatenate(B)

def train_one(feat, motion, y, base, use_motion, epochs, device, pos_weight, seed=0):
    torch.manual_seed(seed)
    head=ResidualHead(use_motion=use_motion).to(device)
    opt=torch.optim.Adam(head.parameters(),lr=1e-3,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    F=torch.tensor(feat,dtype=torch.float32); M=torch.tensor(motion,dtype=torch.float32)
    Y=torch.tensor(y,dtype=torch.float32); BL=inv_sigmoid(torch.tensor(base,dtype=torch.float32))
    n=F.shape[0]; bs=4096
    for ep in range(epochs):
        perm=torch.randperm(n); tot=0.0; head.train()
        for i in range(0,n,bs):
            idx=perm[i:i+bs]
            fb=F[idx].to(device); yb=Y[idx].to(device); bb=BL[idx].to(device)
            mb=M[idx].to(device) if use_motion else None
            opt.zero_grad(); out=head(bb,fb,mb); loss=lossf(out,yb); loss.backward(); opt.step()
            tot+=loss.item()*len(idx)
        print(f'  [{"C" if use_motion else "B"}] epoch {ep+1}/{epochs} loss {tot/n:.4f}',flush=True)
    return head

def main():
    p=argparse.ArgumentParser()
    p.add_argument('-cache',default=os.path.expanduser('~/cache')); p.add_argument('-epochs',type=int,default=20)
    p.add_argument('-seed',type=int,default=0, help='0 reproduces the reported run')
    p.add_argument('-head_dir',default=None, help='where to save head_B/C.pt (default: -cache)')
    a=p.parse_args()
    head_dir=a.head_dir or a.cache; os.makedirs(head_dir,exist_ok=True)
    dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    feat,motion,y,base=load_split(a.cache,'train')
    print(f'train pairs {feat.shape[0]}, feat {feat.shape[1]}, motion {motion.shape[1]}, contact {y.shape[1]}',flush=True)
    # class-balanced pos_weight = neg/pos per contact class (capped)
    pos=y.sum(0)+1; neg=(y.shape[0]-y.sum(0))+1
    pw=torch.tensor(np.clip(neg/pos, 1.0, 50.0), dtype=torch.float32)
    print('pos_weight (rare predicates get more):', np.round(pw.numpy(),1),flush=True)
    for name,use in [('B',False),('C',True)]:
        head=train_one(feat,motion,y,base,use,a.epochs,dev,pw,a.seed)
        torch.save({'state_dict':head.state_dict(),'use_motion':use}, os.path.join(head_dir,f'head_{name}.pt'))
        print(f'saved head_{name}.pt',flush=True)

if __name__=='__main__': main()
