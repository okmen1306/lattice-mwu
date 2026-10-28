# -*- coding: utf-8 -*-
"""DENEY 290: UYGULAMA-KAOS BANDI -- 1 puanlik farklarimiz olculebilir mi?

SUPHE: D-287 MNIST'te 0.8094, D-289 ayni konfigurasyonda 0.7916 verdi (fark 1.78!).
Tek fark LUT ifadesi: 2.0**k  vs  exp(k*ln2)  -- matematiksel olarak OZDES,
sayisal olarak 1e-15 farkli. Ayni sey 20NG'de de var: bizim motor D-285'te 0.9062,
D-286'da 0.9045, D-289'da 0.9138 verdi (aralik 0.93 puan).

Eger dogruysa, ~1 puanlik TUM kiyaslarimiz UYGULAMA GURULTUSUNUN icinde kaliyor
ve tek kosuda olculemez. Bu, [[BULGU_Cevrimici_Ogrenme_Kaotiktir]]'in genisletilmesidir.

TEST: ayni tohum, ayni her sey, SADECE LUT ifadesi degisik. 6 tohum.
"""
import math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import metrics
import deney_283_coklu_veri as DS
LN2 = math.log(2.0); SEEDS = list(range(6))

def run_eng(rows, ys, D, B, k_res, lutmode, order):
    S = 2**(B-1)-1
    kk = np.arange(-S, S+1, dtype=np.int64); mag = np.abs(kk).astype(np.float64)
    core = (2.0**np.maximum(mag-1,0)) if lutmode=="pow2" else np.exp(np.maximum(mag-1,0)*LN2)
    lut = np.sign(kk)*np.where(mag==0, 0.0, core)
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

print("="*92); print("### D-290: UYGULAMA-KAOS BANDI (ayni matematik, iki LUT ifadesi)"); print("="*92)
print("LUT farki: 2.0**3 = %.20g   exp(3*ln2) = %.20g   fark = %.3g"
      % (2.0**3, math.exp(3*LN2), 2.0**3 - math.exp(3*LN2)))
for nm, loader in [("MNIST 0-4 vs 5-9", DS.mnist), ("20NG comp vs rec", DS.LOADERS[0])]:
    _, rtr, ytr, rte, yte, D = loader()
    A, Bc = [], []
    for seed in SEEDS:
        rng = np.random.RandomState(seed); o = rng.permutation(len(rtr))
        tro = o[int(0.2*len(o)):]
        A.append(metrics(pr(run_eng(rtr,ytr,D,4,0,"pow2",tro), rte), yte)[0])
        Bc.append(metrics(pr(run_eng(rtr,ytr,D,4,0,"exp",tro), rte), yte)[0])
    A, Bc = np.array(A), np.array(Bc)
    d = A-Bc; sd = d.std(ddof=1)
    t = d.mean()/(sd/math.sqrt(len(d))) if sd>0 else float("nan")
    print("\n%s  (B=4, k=0, koordinat programi, 6 tohum)" % nm)
    print("  LUT=2.0**k    : %.4f +/-%.4f   tohumlar: %s" % (A.mean(), A.std(), np.round(A,4).tolist()))
    print("  LUT=exp(k*ln2): %.4f +/-%.4f   tohumlar: %s" % (Bc.mean(), Bc.std(), np.round(Bc,4).tolist()))
    print("  >> OZDES iki uygulama arasi fark: %+.2f puan (t=%.2f)" % (d.mean()*100, t))
    print("  >> tohum basina max |fark|: %.2f puan" % (np.abs(d).max()*100))
    print("  >> KAOS BANDI (tohum-ici max sapma): %.2f puan" % (max(A.max()-A.min(), Bc.max()-Bc.min())*100))
