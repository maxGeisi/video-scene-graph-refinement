import numpy as np, glob, os
f=sorted(glob.glob(os.path.expanduser('~/cache/train_*.npz')))[0]
d=np.load(f)
y=d['y']; base=d['base']
print("y shape:", y.shape, "dtype:", y.dtype)
print("y ndim:", y.ndim)
print("first 5 y rows:")
print(y[:5])
print("y min:", y.min(), "max:", y.max())
if y.ndim==1:
    print("unique y values (first 20):", np.unique(y)[:20])
    print("value counts of each class:")
    u,c=np.unique(y,return_counts=True)
    for uu,cc in zip(u,c): print(f"  class {uu}: {cc}")
else:
    print("y is multi-dim; column sums (how many positives per class):")
    print(y.sum(axis=0))
print("\nbase shape:", base.shape, "base[0]:", np.round(base[0],3))
