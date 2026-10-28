# -*- coding: utf-8 -*-
"""D277 -- IKI KONTROL

A) %90.76 vs %90.44 SAHTE MI?
   D275'te tek ray 4 bit = %90.44 olcuLdu. D276'da AYNI algoritma %90.76 verdi.
   Tek fark: LUT'un nasil hesaplandigi.
       D275: lut[i] = exp(i * ln2)   -> 8 yerine 7.999999999999998 cikabilir
       D276: lut[i] = 2.0 ** i      -> tam 8
   Matematiksel olarak AYNI. Eger sonuclar farkliysa, fark ALGORITMIK degil,
   [[BULGU_Cevrimici_Ogrenme_Kaotiktir]]'daki kayan nokta kaosudur; yani
   "%90.76 ile FP64'u gectik" iddiasi D273'un yaptigi hatanin birebir tekraridir.

B) 7 BIT + STOKASTIK: Warmuth'a kalan -2.22 puanlik acik kapaniyor mu?
"""
import math, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import run_mwu, pred_mwu, make_synth, load_real, LN2, ETA_GRID, metrics
from deney_275_tek_ray import run_signed, pred_signed
from deney_276_stokastik_kafes import run_signed_p

SEEDS = [0, 1, 2, 3, 4]


def split(rtr, seed):
    r = np.random.RandomState(seed)
    o = r.permutation(len(rtr))
    nv = int(0.2 * len(o))
    return o[:nv], o[nv:]


def warmuth_ref(rtr, ytr, rte, yte, D, vi, tro):
    rval, yval = [rtr[i] for i in vi], ytr[vi]
    be, bv = None, -1.0
    for eta in ETA_GRID:
        m, _ = run_mwu(rtr, ytr, D, eta, tro)
        v = float((pred_mwu(m, rval) == yval).mean())
        if v > bv:
            bv, be = v, eta
    m, _ = run_mwu(rtr, ytr, D, be, tro)
    return metrics(pred_mwu(m, rte), yte)[0]


def run_signed_pow2(rows, ys, D, B_exp, order):
    """D276'nin LUT'u: lut[i] = 2.0**i (tam kuvvet)."""
    g = np.random.default_rng(0)
    return run_signed_p(rows, ys, D, B_exp, 1.0, order, g)


print("=" * 84, flush=True)
print("KONTROL A -- GERCEK VERI, tek ray 4 bit, IKI ESDEGER LUT", flush=True)
print("=" * 84, flush=True)
a_exp, a_pow, a_ref = [], [], []
for seed in SEEDS:
    rtr, ytr, rte, yte, D = load_real(seed)
    vi, tro = split(rtr, seed)
    a_ref.append(warmuth_ref(rtr, ytr, rte, yte, D, vi, tro))
    m, _ = run_signed(rtr, ytr, D, 3, LN2, tro)          # lut = exp(i*ln2)
    a_exp.append(metrics(pred_signed(m, rte), yte)[0])
    m, _ = run_signed_pow2(rtr, ytr, D, 3, tro)          # lut = 2.0**i
    a_pow.append(metrics(pred_signed(m, rte), yte)[0])
    print("  seed %d: exp-LUT %.2f | pow2-LUT %.2f | Warmuth %.2f"
          % (seed, a_exp[-1] * 100, a_pow[-1] * 100, a_ref[-1] * 100), flush=True)

E, P, R = np.array(a_exp), np.array(a_pow), np.array(a_ref)
print("\n  LUT = exp(i*ln2) : %.2f%% +/-%.2f" % (E.mean() * 100, E.std() * 100))
print("  LUT = 2.0**i     : %.2f%% +/-%.2f" % (P.mean() * 100, P.std() * 100))
print("  Warmuth FP64     : %.2f%% +/-%.2f" % (R.mean() * 100, R.std() * 100))
d = P - E
sd = d.std(ddof=1)
t = d.mean() / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
print("\n  >> MATEMATIKSEL OLARAK AYNI iki uygulama arasindaki fark: %+.2f puan (t=%.2f)"
      % (d.mean() * 100, t))
print("  >> LUT degerleri esit mi? %s"
      % str(np.allclose([math.exp(i * LN2) for i in range(7)],
                        [2.0 ** i for i in range(7)])))
print("  >> Tam esit mi (bit bit)? %s"
      % str([math.exp(i * LN2) for i in range(7)] == [2.0 ** i for i in range(7)]))
for i in range(7):
    x, y2 = math.exp(i * LN2), 2.0 ** i
    if x != y2:
        print("     i=%d: exp=%.20g  pow2=%.20g  fark=%.3g" % (i, x, y2, x - y2))

print("\n" + "=" * 84, flush=True)
print("KONTROL B -- YOGUN SENTETIK, tek ray 7 bit + stokastik adim", flush=True)
print("=" * 84, flush=True)
P_GRID = [1.0, 0.5, 0.25, 0.1]
res, ref2 = {}, []
for seed in SEEDS:
    rtr, ytr, rte, yte, D = make_synth(seed)
    vi, tro = split(rtr, seed)
    ref2.append(warmuth_ref(rtr, ytr, rte, yte, D, vi, tro))
    for p in P_GRID:
        g = np.random.default_rng(30000 + seed)
        m, _ = run_signed_p(rtr, ytr, D, 6, p, tro, g)
        res.setdefault(p, []).append(metrics(pred_signed(m, rte), yte)[0])
    print("  seed %d bitti" % seed, flush=True)

R2 = np.array(ref2)
print("\n  Warmuth FP64 (eta* ayarli): %.2f%% +/-%.2f" % (R2.mean() * 100, R2.std() * 100))
for p in P_GRID:
    a = np.array(res[p])
    d = a - R2
    sd = d.std(ddof=1)
    t = d.mean() / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
    mark = "  <-- BERABERE" if abs(t) < 2.776 else ""
    print("  7 bit tek ray, p=%-5g : %.2f%% +/-%.2f  (Warmuth'a %+.2f, t=%.2f)%s"
          % (p, a.mean() * 100, a.std() * 100, d.mean() * 100, t, mark))
