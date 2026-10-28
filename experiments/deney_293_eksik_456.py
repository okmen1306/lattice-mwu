# -*- coding: utf-8 -*-
"""D-293: EKSIK 4, 5, 6 -- tek kosuda

(4) n_i SAYACININ BIT MALIYETI: program icin n_i tutuyoruz ama bit butcesine
    yazmadik. Aslinda sadece shift=floor(log2 n)//2 lazim (0..5 arasi = 3 bit).
    n_i'yi b bitlik DOYUMLU sayacla degistir, b in {3,4,6,8,inf} tara.
(5) KAOS BANDI RCV1 ve SENTETIKTE: D-290'i o iki tezgahta tekrarla.
(6) IKILI OLMAYAN OZELLIK: 20NG'yi TF-IDF (gercek degerli) ile kosur.
    Motor: r_i += d*round(8*x_i)  -> hala TAMSAYI. Kiyas: ayni ozelliklerle MWU.
"""
import math, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import metrics, make_synth, to_rows
from deney_285_adil_dovus import mwu_sched, pred_mwu2
import deney_283_coklu_veri as DS
LN2=math.log(2.0)

def eng(rows, ys, D, B, k_res, order, nbits=None, xvals=None):
    S=2**(B-1)-1
    kk=np.arange(-S,S+1,dtype=np.int64); m=np.abs(kk).astype(np.float64)
    lut=np.sign(kk)*np.where(m==0,0.0,2.0**np.maximum(m-1,0))
    s=np.zeros(D,np.int64); r=np.zeros(D,np.int64); n=np.zeros(D,np.int64)
    NMAX = (2**nbits - 1) if nbits else None
    for t in order:
        a=rows[t]; y=ys[t]
        if len(a)==0: continue
        w=lut[s[a]+S]
        sc=float((w*xvals[t]).sum()) if xvals is not None else float(w.sum())
        if (1 if sc>0 else 0)!=y:
            n[a]=np.minimum(n[a]+1, NMAX) if NMAX else n[a]+1
            sh=(np.floor(np.log2(np.maximum(n[a],1))).astype(np.int64))//2
            thr=np.left_shift(np.int64(2)**k_res,sh)
            inc = np.rint(8*xvals[t]).astype(np.int64) if xvals is not None else 1
            r[a]+= (1 if y==1 else -1)*inc; ra=r[a]
            st=np.where(ra>=thr,1,np.where(ra<=-thr,-1,0))
            r[a]=ra-st*thr; s[a]=np.clip(s[a]+st,-S,S)
    return lut,s,S

def pr(mo,rows,xv=None):
    lut,s,S=mo; out=np.zeros(len(rows),np.int8)
    for i,a in enumerate(rows):
        if len(a):
            w=lut[s[a]+S]
            out[i]=1 if float((w*xv[i]).sum() if xv is not None else w.sum())>0 else 0
    return out

print("="*90, flush=True); print("### D-293: EKSIK 4, 5, 6", flush=True); print("="*90, flush=True)

# ---------- EKSIK 4 ----------
print("\n--- (4) n_i SAYACININ BIT MALIYETI (20NG + MNIST, 5 tohum) ---", flush=True)
for nm, ld in [("20NG", lambda: DS.LOADERS[0]()), ("MNIST", DS.mnist)]:
    _,rtr,ytr,rte,yte,D = ld()
    out={}
    for seed in range(5):
        rg=np.random.RandomState(seed); o=rg.permutation(len(rtr)); tro=o[int(0.2*len(o)):]
        for b in [3,4,6,8,None]:
            mo=eng(rtr,ytr,D,4,0,tro,nbits=b)
            out.setdefault(b,[]).append(metrics(pr(mo,rte),yte)[0])
    ref=np.array(out[None])
    print("  %s:"%nm, flush=True)
    for b in [3,4,6,8,None]:
        a=np.array(out[b]); d=a-ref; sd=d.std(ddof=1)
        t=d.mean()/(sd/math.sqrt(5)) if sd>0 else 0.0
        print("    n_i = %-8s toplam %-9s %.4f +/-%.4f  sinirsiza fark %+5.2f (t=%5.2f)"
              %("%d bit"%b if b else "sinirsiz", "%d bit"%(4+b) if b else "4+64 bit",
                a.mean(),a.std(),d.mean()*100,t), flush=True)

# ---------- EKSIK 5 ----------
print("\n--- (5) KAOS BANDI: RCV1 ve SENTETIK (ozdes iki LUT, 6 tohum) ---", flush=True)
def band(nm, get):
    A,B_=[],[]
    for seed in range(6):
        rtr,ytr,rte,yte,D=get(seed)
        rg=np.random.RandomState(seed); o=rg.permutation(len(rtr)); tro=o[int(0.2*len(o)):]
        for tag,arr in [("pow2",A),("exp",B_)]:
            S=2**3-1; kk=np.arange(-S,S+1,dtype=np.int64); m=np.abs(kk).astype(np.float64)
            core=(2.0**np.maximum(m-1,0)) if tag=="pow2" else np.exp(np.maximum(m-1,0)*LN2)
            lut=np.sign(kk)*np.where(m==0,0.0,core)
            s=np.zeros(D,np.int64); r=np.zeros(D,np.int64); n=np.zeros(D,np.int64)
            for t in tro:
                a=rtr[t]; y=ytr[t]
                if len(a)==0: continue
                if (1 if float(lut[s[a]+S].sum())>0 else 0)!=y:
                    n[a]+=1; sh=(np.floor(np.log2(np.maximum(n[a],1))).astype(np.int64))//2
                    thr=np.left_shift(np.int64(1),sh)
                    r[a]+=(1 if y==1 else -1); ra=r[a]
                    st=np.where(ra>=thr,1,np.where(ra<=-thr,-1,0))
                    r[a]=ra-st*thr; s[a]=np.clip(s[a]+st,-S,S)
            arr.append(metrics(np.array([1 if (len(a) and float(lut[s[a]+S].sum())>0) else 0
                                         for a in rte],np.int8), yte)[0])
    A,B_=np.array(A),np.array(B_); d=A-B_
    print("  %-10s ozdes iki uygulama: %+.2f puan | tohum basi max |fark| %.2f | KAOS BANDI %.2f"
          %(nm, d.mean()*100, np.abs(d).max()*100, max(A.max()-A.min(),B_.max()-B_.min())*100), flush=True)
band("RCV1", lambda s: DS.rcv1()[1:])
band("SENTETIK", make_synth)

# ---------- EKSIK 6 ----------
print("\n--- (6) IKILI OLMAYAN OZELLIK: 20NG TF-IDF (gercek degerli x) ---", flush=True)
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
COMP=["comp.graphics","comp.os.ms-windows.misc","comp.sys.ibm.pc.hardware","comp.sys.mac.hardware","comp.windows.x"]
REC=["rec.autos","rec.motorcycles","rec.sport.baseball","rec.sport.hockey"]
tr=fetch_20newsgroups(subset="train",categories=COMP+REC,remove=("headers","footers","quotes"),shuffle=False)
te=fetch_20newsgroups(subset="test",categories=COMP+REC,remove=("headers","footers","quotes"),shuffle=False)
lab=lambda t: np.array([1 if t.target_names[i] in COMP else 0 for i in t.target],np.int8)
v=TfidfVectorizer(max_features=10000,min_df=2,stop_words="english",sublinear_tf=True)
Xtr=v.fit_transform(tr.data); Xte=v.transform(te.data)
rtr,ytr=to_rows(Xtr),lab(tr); rte,yte=to_rows(Xte),lab(te)
Xtr=sparse.csr_matrix(Xtr); Xte=sparse.csr_matrix(Xte)
vtr=[Xtr.data[Xtr.indptr[i]:Xtr.indptr[i+1]] for i in range(Xtr.shape[0])]
vte=[Xte.data[Xte.indptr[i]:Xte.indptr[i+1]] for i in range(Xte.shape[0])]
D=Xtr.shape[1]
ours,w=[],[]
for seed in range(5):
    rg=np.random.RandomState(seed); o=rg.permutation(len(rtr)); nv=int(0.2*len(o))
    vi,tro=o[:nv],o[nv:]; rv,yv=[rtr[i] for i in vi],ytr[vi]; vv=[vtr[i] for i in vi]
    bb,bv=None,-1.0
    for c in [0.5,1.0,2.0,4.0]:
        m,_=mwu_sched(rtr,ytr,D,c,"koord",tro); val=float((pred_mwu2(m,rv)==yv).mean())
        if val>bv: bv,bb=val,c
    m,_=mwu_sched(rtr,ytr,D,bb,"koord",tro); w.append(metrics(pred_mwu2(m,rte),yte)[0])
    bk,bkv=None,-1.0
    for k_ in [0,2]:
        mo=eng(rtr,ytr,D,4,k_,tro,xvals=vtr); val=float((pr(mo,rv,vv)==yv).mean())
        if val>bkv: bkv,bk=val,k_
    mo=eng(rtr,ytr,D,4,bk,tro,xvals=vtr)
    ours.append(metrics(pr(mo,rte,vte),yte)[0])
O,W=np.array(ours),np.array(w); d=O-W; sd=d.std(ddof=1); t=d.mean()/(sd/math.sqrt(5))
print("  MWU (gercek deger, 64 bit): %.4f +/-%.4f"%(W.mean(),W.std()), flush=True)
print("  BIZ  (4+k bit, tamsayi)   : %.4f +/-%.4f   fark %+.2f (t=%.2f)%s"
      %(O.mean(),O.std(),d.mean()*100,t,"  BERABERE" if abs(t)<2.776 else ""), flush=True)
