# -*- coding: utf-8 -*-
"""D-291 (EKSIK 1): MNIST 25 TOHUM.
D-290: MNIST'te eslesmis fark sd=1.57 puan -> 5 tohumda MDE=1.95, iddia edilen
acik 1.7 -> OLCULEMEZ. 25 tohumda MDE = 2.776*1.57/5 = 0.87 puan -> olculebilir.
"""
import math, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import metrics
from deney_285_adil_dovus import mwu_sched, pred_mwu2
import deney_283_coklu_veri as DS
LN2 = math.log(2.0); SEEDS = list(range(25))
ETA = [0.025, 0.1, 0.25, 0.5, 1.0]; CG = [0.5, 1.0, 2.0, 4.0]

def eng(rows, ys, D, B, k_res, order):
    S = 2**(B-1)-1
    kk = np.arange(-S, S+1, dtype=np.int64); mg = np.abs(kk).astype(np.float64)
    lut = np.sign(kk)*np.where(mg==0, 0.0, 2.0**np.maximum(mg-1,0))
    s = np.zeros(D, np.int64); r = np.zeros(D, np.int64); n = np.zeros(D, np.int64)
    for t in order:
        a = rows[t]; y = ys[t]
        sc = float(lut[s[a]+S].sum()) if len(a) else 0.0
        if (1 if sc>0 else 0) != y and len(a):
            n[a] += 1
            sh = (np.floor(np.log2(np.maximum(n[a],1))).astype(np.int64))//2
            thr = np.left_shift(np.int64(2)**k_res, sh)
            r[a] += (1 if y==1 else -1); ra = r[a]
            st = np.where(ra>=thr,1,np.where(ra<=-thr,-1,0))
            r[a] = ra - st*thr; s[a] = np.clip(s[a]+st, -S, S)
    return lut, s, S

def pr(mo, rows):
    lut,s,S = mo
    return np.array([1 if (len(a) and float(lut[s[a]+S].sum())>0) else 0 for a in rows], np.int8)

print("="*88, flush=True); print("### D-291 (EKSIK 1): MNIST, 25 TOHUM", flush=True); print("="*88, flush=True)
_, rtr, ytr, rte, yte, D = DS.mnist()
res = {}
t0=time.time()
for seed in SEEDS:
    rg = np.random.RandomState(seed); o = rg.permutation(len(rtr)); nv=int(0.2*len(o))
    vi, tro = o[:nv], o[nv:]; rv, yv = [rtr[i] for i in vi], ytr[vi]
    for lbl, mode, grid in [("W1 Warmuth sabit","sabit",ETA),("W2 Warmuth c/sqrt(t)","kuresel",CG),
                            ("W3 MWU+AdaGrad","koord",CG)]:
        bb,bv=None,-1.0
        for c in grid:
            m,_=mwu_sched(rtr,ytr,D,c,mode,tro); v=float((pred_mwu2(m,rv)==yv).mean())
            if v>bv: bv,bb=v,c
        m,_=mwu_sched(rtr,ytr,D,bb,mode,tro)
        res.setdefault(lbl,[]).append(metrics(pred_mwu2(m,rte),yte)[0])
    bk,bv=None,-1.0
    for k_ in [0,2]:
        mo=eng(rtr,ytr,D,4,k_,tro); v=float((pr(mo,rv)==yv).mean())
        if v>bv: bv,bk=v,k_
    mo=eng(rtr,ytr,D,4,bk,tro)
    res.setdefault("BIZ 4+k bit",[]).append(metrics(pr(mo,rte),yte)[0])
    if (seed+1)%5==0: print("  %d/25 tohum (%.1f dk)"%(seed+1,(time.time()-t0)/60), flush=True)

ours = np.array(res["BIZ 4+k bit"]); n=len(SEEDS)
print("\n  %-22s %9s %9s   %s"%("model","acc","std","bize gore (25 tohum, t_krit=2.06)"), flush=True)
for lbl in ["W1 Warmuth sabit","W2 Warmuth c/sqrt(t)","W3 MWU+AdaGrad","BIZ 4+k bit"]:
    a=np.array(res[lbl]); tag=""
    if not lbl.startswith("BIZ"):
        d=ours-a; sd=d.std(ddof=1); t=d.mean()/(sd/math.sqrt(n))
        mde = 2.06*sd/math.sqrt(n)
        tag="%+6.2f (t=%6.2f)%s  MDE=%.2f"%(d.mean()*100,t,
            "  BERABERE" if abs(t)<2.06 else ("  BIZ KAZANDIK" if t>0 else "  BIZ KAYBETTIK"), mde*100)
    print("  %-22s %9.4f %9.4f   %s"%(lbl,a.mean(),a.std(),tag), flush=True)
