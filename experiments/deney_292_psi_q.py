# -*- coding: utf-8 -*-
"""D-292 (EKSIK 3): PSI_q BIRLESIK FORMULU -- ilk kez test ediliyor.

Psi_q(s) = sgn(s)[ (1+(1-q)*delta*|s|)^(1/(1-q)) - 1 ]
  q=0    -> delta*|s|          (dogrusal)      = OKLID canagi = tamsayi Perceptron
  0<q<1  -> ~ |s|^(1/(1-q))    (kuvvet)        = p-NORM canagi
  q->1   -> e^(delta|s|) - 1   (ustel)         = ENTROPI canagi = bit-shift
Psi_q(0)=0 -> olu sinaps her q'da korunur.
Tek sayac, tek guncelleme kurali; q SADECE LUT'u degistiriyor.

SORU: D-279'da hicbir tezgah UC NOKTA secmemisti (seyrek r~1-2, yogun r=6).
Ic nokta gercekten iki ucu da geciyor mu?
"""
import math, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import metrics, make_synth
from deney_285_adil_dovus import mwu_sched, pred_mwu2
import deney_283_coklu_veri as DS
LN2 = math.log(2.0); SEEDS=[0,1,2,3,4]
Q_GRID = [0.0, 0.25, 0.5, 0.667, 0.75, 0.875, 0.999]   # 0.999 ~ entropi

def psi(B, q, delta=LN2):
    S = 2**(B-1)-1
    kk = np.arange(-S,S+1,dtype=np.int64); m = np.abs(kk).astype(np.float64)
    if q >= 0.999:
        core = np.exp(delta*m) - 1.0
    else:
        core = np.power(1.0+(1.0-q)*delta*m, 1.0/(1.0-q)) - 1.0
    return np.sign(kk)*core, S

def eng(rows, ys, D, B, q, k_res, order):
    lut, S = psi(B, q)
    s=np.zeros(D,np.int64); r=np.zeros(D,np.int64); n=np.zeros(D,np.int64)
    for t in order:
        a=rows[t]; y=ys[t]
        sc=float(lut[s[a]+S].sum()) if len(a) else 0.0
        if (1 if sc>0 else 0)!=y and len(a):
            n[a]+=1
            sh=(np.floor(np.log2(np.maximum(n[a],1))).astype(np.int64))//2
            thr=np.left_shift(np.int64(2)**k_res, sh)
            r[a]+=(1 if y==1 else -1); ra=r[a]
            st=np.where(ra>=thr,1,np.where(ra<=-thr,-1,0))
            r[a]=ra-st*thr; s[a]=np.clip(s[a]+st,-S,S)
    return lut,s,S

def pr(mo,rows):
    lut,s,S=mo
    return np.array([1 if (len(a) and float(lut[s[a]+S].sum())>0) else 0 for a in rows],np.int8)

def study(name, get):
    print("\n"+"="*90, flush=True); print("### %s"%name, flush=True); print("="*90, flush=True)
    res, w3 = {}, []
    for seed in SEEDS:
        rtr,ytr,rte,yte,D = get(seed)
        rg=np.random.RandomState(seed); o=rg.permutation(len(rtr)); nv=int(0.2*len(o))
        vi,tro=o[:nv],o[nv:]; rv,yv=[rtr[i] for i in vi],ytr[vi]
        bb,bv=None,-1.0
        for c in [0.5,1.0,2.0,4.0]:
            m,_=mwu_sched(rtr,ytr,D,c,"koord",tro); v=float((pred_mwu2(m,rv)==yv).mean())
            if v>bv: bv,bb=v,c
        m,_=mwu_sched(rtr,ytr,D,bb,"koord",tro); w3.append(metrics(pred_mwu2(m,rte),yte)[0])
        for q in Q_GRID:
            bk,bkv=None,-1.0
            for k_ in [0,2]:
                mo=eng(rtr,ytr,D,6,q,k_,tro); v=float((pr(mo,rv)==yv).mean())
                if v>bkv: bkv,bk=v,k_
            mo=eng(rtr,ytr,D,6,q,bk,tro)
            res.setdefault(q,[]).append(metrics(pr(mo,rte),yte)[0])
        print("  seed %d"%seed, flush=True)
    W=np.array(w3)
    print("\n  W3 referans: %.4f +/-%.4f"%(W.mean(),W.std()), flush=True)
    best=max(Q_GRID,key=lambda q: np.mean(res[q]))
    for q in Q_GRID:
        a=np.array(res[q]); d=a-W; sd=d.std(ddof=1)
        t=d.mean()/(sd/math.sqrt(len(d))) if sd>0 else float("nan")
        lbl = "q=%.3f %s"%(q, "(OKLID)" if q==0 else ("(ENTROPI)" if q>=0.999 else "(p-norm)"))
        print("  %-22s %.4f +/-%.4f   W3'e %+6.2f (t=%6.2f)%s%s"
              %(lbl,a.mean(),a.std(),d.mean()*100,t,
                "  BERABERE" if abs(t)<2.776 else "", "   <== EN IYI" if q==best else ""), flush=True)
    e0=np.mean(res[0.0]); e1=np.mean(res[0.999]); eb=np.mean(res[best])
    print("  >> ic nokta iki ucu geciyor mu? en iyi %.4f | Oklid %.4f | Entropi %.4f -> %s"
          %(eb,e0,e1, "EVET" if (eb>e0 and eb>e1) else "HAYIR"), flush=True)

t0=time.time()
study("20NG comp vs rec (seyrek)", lambda s: DS.LOADERS[0]()[1:])
study("YOGUN SENTETIK", make_synth)
study("MNIST", lambda s: DS.mnist()[1:])
print("\ntoplam %.1f dk"%((time.time()-t0)/60), flush=True)
