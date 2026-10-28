# -*- coding: utf-8 -*-
"""DENEY 289: MNIST -- KAFESI INCELT (olcumden turetilmis, tahmin degil)

D-288 TESHISI: +-127 mevcutken sadece +-4 kullaniliyor; |s| dagilimi
[411,281,56,28,8,0,...]. Yani MENZIL degil, COZUNURLUK eksik. Model 1 ile 2,
2 ile 4 arasinda agirlik istiyor; entropi kafesi ona sadece 2x adim veriyor.

DOGRUDAN SONUC: kafesi incelt -> delta = ln2/m, agirlik = 2^(s/m).
  m=1 : mevcut (saf bit-shift)
  m=2 : sqrt(2)'nin kuvvetleri -> 2x ince
  m=4 : 4x ince
  m=8 : 8x ince
Saf `1<<k` bozulur ama TABLO OKUMASI ZATEN TABLO OKUMASI -- donanim maliyeti ayni.

Bu bir mekanizma hipotezi DEGIL; olculmus bir eksigin (kullanilan 9 seviye)
dogrudan karsiligi.
"""
import math, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import metrics
from deney_285_adil_dovus import mwu_sched, pred_mwu2, C_GRID
import deney_283_coklu_veri as DS

LN2 = math.log(2.0); SEEDS = [0,1,2]

def run_eng(rows, ys, D, B, k_res, m, order):
    S = 2**(B-1)-1; delta = LN2/m
    kk = np.arange(-S, S+1, dtype=np.int64); mag = np.abs(kk).astype(np.float64)
    lut = np.sign(kk)*np.where(mag==0, 0.0, np.exp(np.maximum(mag-1,0)*delta))
    s = np.zeros(D, np.int64); res = np.zeros(D, np.int64); n = np.zeros(D, np.int64)
    for t in order:
        a = rows[t]; y = ys[t]
        sc = float(lut[s[a]+S].sum()) if len(a) else 0.0
        if (1 if sc>0 else 0) != y and len(a):
            n[a] += 1
            sh = (np.floor(np.log2(np.maximum(n[a],1))).astype(np.int64))//2
            thr = np.left_shift(np.int64(2)**k_res, sh)
            res[a] += (1 if y==1 else -1); ra = res[a]
            stp = np.where(ra>=thr, 1, np.where(ra<=-thr, -1, 0))
            res[a] = ra - stp*thr
            s[a] = np.clip(s[a]+stp, -S, S)
    return (lut, s, S)

def pr(mo, rows):
    lut, s, S = mo
    return np.array([1 if (len(a) and float(lut[s[a]+S].sum())>0) else 0 for a in rows], np.int8)

def study(name, loader):
    print("\n"+"="*94, flush=True); print("### %s" % name, flush=True); print("="*94, flush=True)
    _, rtr, ytr, rte, yte, D = loader()
    res, w3, cfg = {}, [], {}
    for seed in SEEDS:
        rng = np.random.RandomState(seed); o = rng.permutation(len(rtr)); nv = int(0.2*len(o))
        vi, tro = o[:nv], o[nv:]; rval, yval = [rtr[i] for i in vi], ytr[vi]
        bb, bv = None, -1.0
        for c in C_GRID:
            mm,_ = mwu_sched(rtr, ytr, D, c, "koord", tro)
            v = float((pred_mwu2(mm, rval)==yval).mean())
            if v>bv: bv, bb = v, c
        mm,_ = mwu_sched(rtr, ytr, D, bb, "koord", tro)
        w3.append(metrics(pred_mwu2(mm, rte), yte)[0])
        for m in [1,2,4,8]:
            best, bvv = None, -1.0
            for B in [4,6]:
                for k_ in [0,2]:
                    mo = run_eng(rtr, ytr, D, B, k_, m, tro)
                    v = float((pr(mo, rval)==yval).mean())
                    if v>bvv: bvv, best = v, (B,k_)
            mo = run_eng(rtr, ytr, D, best[0], best[1], m, tro)
            res.setdefault(m, []).append(metrics(pr(mo, rte), yte)[0])
            cfg.setdefault(m, []).append("B=%d,k=%d" % best)
        print("  seed %d bitti" % seed, flush=True)
    W = np.array(w3)
    print("\n  W3 (MWU+AdaGrad, 64 bit): %.4f +/-%.4f\n" % (W.mean(), W.std()), flush=True)
    for m in [1,2,4,8]:
        a = np.array(res[m]); d = a-W; sd = d.std(ddof=1)
        t = d.mean()/(sd/math.sqrt(len(d))) if sd>0 else float("nan")
        mark = "  BERABERE" if abs(t)<4.303 else ("  KAZANDIK" if t>0 else "  kaybettik")
        print("  delta=ln2/%-2d (%dx ince)  %.4f +/-%.4f   %+6.2f (t=%6.2f)%s   %s"
              % (m, m, a.mean(), a.std(), d.mean()*100, t, mark, cfg[m]), flush=True)

t0=time.time()
study("MNIST 0-4 vs 5-9", DS.mnist)
print("  (%.1f dk)" % ((time.time()-t0)/60), flush=True)
study("KONTROL: 20NG comp vs rec (seyrekte incelme gerekmemeli)", DS.LOADERS[0])
