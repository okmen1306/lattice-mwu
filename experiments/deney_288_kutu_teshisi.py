# -*- coding: utf-8 -*-
"""DENEY 288: MNIST -- KUTU KULLANIMI TESHISI + PROGRAM SERTLIGI

D-287: B=4..10 hepsi BIREBIR ayni (0.8094). Bu kapasite siniri degil, imza.
IDDIA: koordinat programi esigi 2^(k+shift) yapiyor; MNIST'te n_i~600 ->
shift=floor(log2 600)//2=4 -> esik 16. Sayaci bir basamak oynatmak icin 16
birikmis adim gerekiyor -> KUTUYA HIC DEGMIYORUZ.

(a) TESHIS: son |s| dagilimi. Kutunun ne kadari kullaniliyor?
(b) DOGRUDAN SONUC: program sertligi (shift bolen) taranir.
    bolen=2 -> p~1/sqrt(n) (mevcut) | 3 -> n^(-1/3) | 4 -> n^(-1/4) | 0 -> program YOK
"""
import math, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import metrics
from deney_285_adil_dovus import mwu_sched, pred_mwu2, C_GRID
import deney_283_coklu_veri as DS

SEEDS = [0, 1, 2]

def run_eng(rows, ys, D, B, k_res, div, order):
    S = 2**(B-1)-1
    kk = np.arange(-S, S+1, dtype=np.int64); mag = np.abs(kk).astype(np.float64)
    lut = np.sign(kk)*np.where(mag==0, 0.0, 2.0**np.maximum(mag-1,0))
    s = np.zeros(D, np.int64); res = np.zeros(D, np.int64); n = np.zeros(D, np.int64)
    for t in order:
        a = rows[t]; y = ys[t]
        sc = float(lut[s[a]+S].sum()) if len(a) else 0.0
        if (1 if sc>0 else 0) != y and len(a):
            n[a] += 1
            sh = np.zeros(len(a), np.int64) if div==0 else \
                 (np.floor(np.log2(np.maximum(n[a],1))).astype(np.int64))//div
            thr = np.left_shift(np.int64(2)**k_res, sh)
            res[a] += (1 if y==1 else -1); ra = res[a]
            stp = np.where(ra>=thr, 1, np.where(ra<=-thr, -1, 0))
            res[a] = ra - stp*thr
            s[a] = np.clip(s[a]+stp, -S, S)
    return (lut, s, S), n

def pr(model, rows):
    lut, s, S = model
    return np.array([1 if (len(a) and float(lut[s[a]+S].sum())>0) else 0 for a in rows], np.int8)

print("="*94, flush=True); print("### D-288: MNIST kutu kullanimi + program sertligi", flush=True); print("="*94, flush=True)
_, rtr, ytr, rte, yte, D = DS.mnist()
rng = np.random.RandomState(0); o = rng.permutation(len(rtr)); tro = o[int(0.2*len(o)):]
print("\n(a) TESHIS -- B=8 kutu (S=127), mevcut program (bolen=2), k=0:", flush=True)
(m, n) = run_eng(rtr, ytr, D, 8, 0, 2, tro)
s = m[1]
print("    max|s| = %d   (kutu siniri S=127)" % np.abs(s).max(), flush=True)
print("    |s|>7 olan sinaps sayisi : %d / %d   <-- 0 ise B=4 zaten yetiyor"
      % (int((np.abs(s)>7).sum()), D), flush=True)
print("    koordinat basina guncelleme n_i: ort %.0f, max %d" % (n.mean(), n.max()), flush=True)
print("    |s| dagilimi: %s" % np.bincount(np.abs(s), minlength=10)[:10].tolist(), flush=True)

print("\n(b) PROGRAM SERTLIGI TARAMASI (B=8, k validation'da)", flush=True)
res, w3 = {}, []
for seed in SEEDS:
    r2 = np.random.RandomState(seed); od = r2.permutation(len(rtr)); nv = int(0.2*len(od))
    vi, tr2 = od[:nv], od[nv:]; rval, yval = [rtr[i] for i in vi], ytr[vi]
    bb, bv = None, -1.0
    for c in C_GRID:
        mm,_ = mwu_sched(rtr, ytr, D, c, "koord", tr2)
        v = float((pred_mwu2(mm, rval)==yval).mean())
        if v>bv: bv, bb = v, c
    mm,_ = mwu_sched(rtr, ytr, D, bb, "koord", tr2)
    w3.append(metrics(pred_mwu2(mm, rte), yte)[0])
    for div in [0, 4, 3, 2]:
        bk, bkv = None, -1.0
        for k_ in [0, 2, 4]:
            mo, _ = run_eng(rtr, ytr, D, 8, k_, div, tr2)
            v = float((pr(mo, rval)==yval).mean())
            if v>bkv: bkv, bk = v, k_
        mo, _ = run_eng(rtr, ytr, D, 8, bk, div, tr2)
        res.setdefault(div, []).append((metrics(pr(mo, rte), yte)[0], bk))
    print("  seed %d bitti" % seed, flush=True)

W = np.array(w3)
print("\n  W3 (MWU+AdaGrad, 64 bit): %.4f +/-%.4f\n" % (W.mean(), W.std()), flush=True)
for div in [0, 4, 3, 2]:
    a = np.array([x[0] for x in res[div]]); ks=[x[1] for x in res[div]]
    d = a-W; sd = d.std(ddof=1); t = d.mean()/(sd/math.sqrt(len(d))) if sd>0 else float("nan")
    lbl = "program YOK" if div==0 else "bolen=%d (p~n^-1/%d)" % (div, div)
    mark = "  BERABERE" if abs(t)<4.303 else ("  KAZANDIK" if t>0 else "  kaybettik")
    print("  %-22s %.4f +/-%.4f   %+6.2f (t=%6.2f)%s   k: %s"
          % (lbl, a.mean(), a.std(), d.mean()*100, t, mark, ks), flush=True)
