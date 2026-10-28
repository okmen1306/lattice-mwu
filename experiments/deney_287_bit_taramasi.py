# -*- coding: utf-8 -*-
"""DENEY 287: MNIST -- BIT GENISLIGI x PROGRAM (hic birlikte denenmedi)

D-283'te B tarandi ama PROGRAMSIZ.  D-284'te program vardi ama B=4 SABIT.
Program MNIST'te dinamigi tamamen degistirdi (+2.5), yani D-283'un B egrisi
artik gecersiz olabilir. Son test edilmemis eksen bu.

B in {4,5,6,7,8,10} x k in {0,2,4}, koordinat programi ACIK.
Kiyas: W3 (MWU+AdaGrad, 64 bit) -- MNIST'te bizi yenen tek sey.
"""
import math, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import metrics
from deney_285_adil_dovus import mwu_sched, pred_mwu2, C_GRID
from deney_286_mnist_canak import run_engine, pred_engine
import deney_283_coklu_veri as DS

SEEDS = [0, 1, 2]
print("=" * 92, flush=True)
print("### D-287: MNIST -- bit genisligi x koordinat programi", flush=True)
print("=" * 92, flush=True)
_, rtr, ytr, rte, yte, D = DS.mnist()
res, w3 = {}, []
for seed in SEEDS:
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(rtr)); nv = int(0.2*len(order))
    vi, tro = order[:nv], order[nv:]
    rval, yval = [rtr[i] for i in vi], ytr[vi]
    bb, bv = None, -1.0
    for c in C_GRID:
        m,_ = mwu_sched(rtr, ytr, D, c, "koord", tro)
        v = float((pred_mwu2(m, rval)==yval).mean())
        if v > bv: bv, bb = v, c
    m,_ = mwu_sched(rtr, ytr, D, bb, "koord", tro)
    w3.append(metrics(pred_mwu2(m, rte), yte)[0])
    for B in [4,5,6,7,8,10]:
        bk, bkv = None, -1.0
        for kk in [0,2,4]:
            mdl,_ = run_engine(rtr, ytr, D, B, "entropi", 0, kk, tro)
            v = float((pred_engine(mdl, rval)==yval).mean())
            if v > bkv: bkv, bk = v, kk
        mdl,_ = run_engine(rtr, ytr, D, B, "entropi", 0, bk, tro)
        res.setdefault(B, []).append((metrics(pred_engine(mdl, rte), yte)[0], bk))
    print("  seed %d bitti" % seed, flush=True)

W = np.array(w3)
print("\n  W3 (MWU+AdaGrad, 64 bit): %.4f +/-%.4f\n" % (W.mean(), W.std()), flush=True)
print("  %-14s %9s %9s   %s" % ("model", "acc", "std", "W3'e fark"), flush=True)
print("  " + "-"*70, flush=True)
for B in [4,5,6,7,8,10]:
    a = np.array([x[0] for x in res[B]]); ks = [x[1] for x in res[B]]
    d = a - W; sd = d.std(ddof=1)
    t = d.mean()/(sd/math.sqrt(len(d))) if sd > 0 else float("nan")
    mark = "  BERABERE" if abs(t) < 4.303 else ("  KAZANDIK" if t>0 else "  kaybettik")
    print("  B=%-12d %9.4f %9.4f   %+6.2f (t=%6.2f)%s   k: %s"
          % (B, a.mean(), a.std(), d.mean()*100, t, mark, ks), flush=True)
