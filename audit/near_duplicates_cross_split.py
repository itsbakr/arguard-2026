# Run 2026-07-29 (UTC). Counts train/dev near-duplicate prompt pairs (whitespace token-set Jaccard >= 0.8). Printed: CROSS 169, label agreement 169/169.
# Body unchanged from the logged run; it reads the public files directly.
import urllib.request, csv, io
import numpy as np
from scipy import sparse
from collections import defaultdict

base="https://raw.githubusercontent.com/araieval/ArGuard-2026-tasks/main/taskB/data/"
def load(f):
    raw=urllib.request.urlopen(base+f).read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(raw)))
tr=load("train.csv"); dv=load("dev_with_label.csv")
rows = [("train",r) for r in tr] + [("dev",r) for r in dv]
N=len(rows); print("N=",N)
split=np.array([0]*len(tr)+[1]*len(dv))
labels=[r['label'] for _,r in rows]
gran=[r['granular_label'] for _,r in rows]

def analyze(tokfn, name, t=0.8):
    sets=[frozenset(tokfn(r['prompt'])) for _,r in rows]
    vocab={}; indptr=[0]; indices=[]
    for s in sets:
        for tok in s:
            indices.append(vocab.setdefault(tok,len(vocab)))
        indptr.append(len(indices))
    X=sparse.csr_matrix((np.ones(len(indices),dtype=np.float32),indices,indptr),shape=(N,len(vocab)))
    sz=np.asarray(X.sum(axis=1)).ravel()
    Xt=X.T.tocsc()
    pairs=[]
    CH=400
    for a in range(0,N,CH):
        b=min(a+CH,N)
        inter=np.asarray((X[a:b] @ Xt).todense())
        union=sz[a:b,None]+sz[None,:]-inter
        with np.errstate(divide='ignore',invalid='ignore'):
            J=np.where(union>0, inter/np.maximum(union,1e-9), 0.0)
        ii,jj=np.nonzero(J>=t-1e-12)
        for i_,j_ in zip(ii,jj):
            gi=a+i_
            if gi<j_: pairs.append((gi,int(j_),float(J[i_,j_])))
    tt=sum(1 for i,j,_ in pairs if split[i]==0 and split[j]==0)
    dd=sum(1 for i,j,_ in pairs if split[i]==1 and split[j]==1)
    cross=[(i,j,s) for i,j,s in pairs if split[i]!=split[j]]
    devrows=set()
    for i,j,_ in cross:
        devrows.add(i if split[i]==1 else j)
    agree_l=sum(1 for i,j,_ in cross if labels[i]==labels[j])
    agree_g=sum(1 for i,j,_ in cross if gran[i]==gran[j])
    print(f"\n=== {name} (t={t}) ===")
    print(f"total pairs>=t : {len(pairs)}   train-train {tt}  dev-dev {dd}  CROSS {len(cross)}")
    print(f"distinct dev rows in cross pairs: {len(devrows)} = {100*len(devrows)/len(dv):.2f}% of {len(dv)}")
    print(f"label agreement on cross pairs   : {agree_l}/{len(cross)} = {100*agree_l/max(len(cross),1):.1f}%")
    print(f"granular agreement on cross pairs: {agree_g}/{len(cross)} = {100*agree_g/max(len(cross),1):.1f}%")
    return pairs, cross, devrows

p1,c1,d1 = analyze(lambda s: s.split(), "whitespace token-set Jaccard")
