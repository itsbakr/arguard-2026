# Run 2026-07-29 (UTC). Same near-duplicate count across tokenizations and Jaccard thresholds 0.6 to 0.8.
# Body unchanged from the logged run; it reads the public files directly.
import urllib.request, csv, io, re, unicodedata
import numpy as np
from scipy import sparse

base="https://raw.githubusercontent.com/araieval/ArGuard-2026-tasks/main/taskB/data/"
def load(f):
    return list(csv.DictReader(io.StringIO(urllib.request.urlopen(base+f).read().decode("utf-8"))))
tr=load("train.csv"); dv=load("dev_with_label.csv")
rows=[("train",r) for r in tr]+[("dev",r) for r in dv]
N=len(rows); split=np.array([0]*len(tr)+[1]*len(dv))
labels=[r['label'] for _,r in rows]; gran=[r['granular_label'] for _,r in rows]
ids=[int(r['id']) for _,r in rows]; prompts=[r['prompt'] for _,r in rows]

def build(sets):
    vocab={}; indptr=[0]; indices=[]
    for s in sets:
        for tok in s: indices.append(vocab.setdefault(tok,len(vocab)))
        indptr.append(len(indices))
    return sparse.csr_matrix((np.ones(len(indices),dtype=np.float32),indices,indptr),shape=(N,len(vocab)))

def pairs_at(tokfn,t):
    sets=[frozenset(tokfn(p)) for p in prompts]
    X=build(sets); sz=np.asarray(X.sum(axis=1)).ravel(); Xt=X.T.tocsc()
    out=[]
    for a in range(0,N,400):
        b=min(a+400,N)
        inter=np.asarray((X[a:b]@Xt).todense())
        union=sz[a:b,None]+sz[None,:]-inter
        J=np.divide(inter,np.maximum(union,1e-9))
        ii,jj=np.nonzero(J>=t-1e-12)
        for i_,j_ in zip(ii,jj):
            gi=a+i_
            if gi<j_: out.append((gi,int(j_),float(J[i_,j_])))
    return out

AR_PUNCT=re.compile(r'[^\w\s]',re.UNICODE)
def norm_ar(s):
    s=AR_PUNCT.sub(' ',s)
    s=re.sub('[ؗ-ًؚ-ْٰـ]','',s)
    s=re.sub('[أإآٱ]','ا',s); s=s.replace('ى','ي').replace('ة','ه')
    return s.split()

toks={
 "A whitespace .split()": lambda s: s.split(),
 "B strip-punct":         lambda s: AR_PUNCT.sub(' ',s).split(),
 "C ar-normalized":       norm_ar,
}
print(f"{'tokenization':24} {'t':5} {'pairs':>6} {'cross':>6} {'devrows':>8} {'%dev':>6} {'lab-agree':>10} {'gran-agree':>10}")
for tn,tf in toks.items():
    for t in (0.8,0.75,0.7,0.6):
        P=pairs_at(tf,t)
        C=[(i,j,s) for i,j,s in P if split[i]!=split[j]]
        dr={ (i if split[i]==1 else j) for i,j,_ in C}
        al=sum(1 for i,j,_ in C if labels[i]==labels[j]); ag=sum(1 for i,j,_ in C if gran[i]==gran[j])
        print(f"{tn:24} {t:<5} {len(P):>6} {len(C):>6} {len(dr):>8} {100*len(dr)/2405:>5.2f}% {al}/{len(C):<8} {ag}/{len(C):<8}")

# id adjacency of cross pairs at 0.8 whitespace
P=pairs_at(lambda s:s.split(),0.8)
C=[(i,j,s) for i,j,s in P if split[i]!=split[j]]
d=[abs(ids[i]-ids[j]) for i,j,_ in C]
d=np.array(d)
print("\nCROSS-pair |id_i - id_j| : min",d.min(),"median",int(np.median(d)),"mean",round(d.mean(),1),"max",d.max())
print("  frac with |delta id| <= 1:",round((d<=1).mean(),3), " <=5:",round((d<=5).mean(),3), " <=50:",round((d<=50).mean(),3))
