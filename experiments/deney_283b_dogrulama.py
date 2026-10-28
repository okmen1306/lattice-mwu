# -*- coding: utf-8 -*-
"""D-283b: IKI BASARISIZLIGI DOGRULA + U-YASASINI GERCEK VERIDE YENIDEN UYDUR

D-283'te 4-bit motor 5 gercek veri setinin 3'unde Warmuth'a yakin kaldi,
IKISINDE geride: sci-vs-talk (-2.37) ve MNIST (-4.48). AMA sadece 3 tohum ve
std yuksekti (0.021 / 0.064). Bu farklar GERCEK mi, gurultu mu?

(a) O iki veri setinde 8 tohumla eslesmis t-testi.
(b) D-283'un bes gercek noktasinda U-yasasini yeniden uydur.
"""
import math, sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import run_mwu, pred_mwu, metrics
from deney_275_tek_ray import run_signed, pred_signed
import deney_283_coklu_veri as DS

LN2 = math.log(2.0)
SEEDS = list(range(8))


def paired(name, loader, B_bits=4):
    _, rtr, ytr, rte, yte, D = loader()
    a4, aw = [], []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(rtr))
        nv = int(0.2 * len(order))
        vi, tro = order[:nv], order[nv:]
        rval, yval = [rtr[i] for i in vi], ytr[vi]
        be, bv = None, -1.0
        for eta in DS.ETA_GRID:
            m, _ = run_mwu(rtr, ytr, D, eta, tro)
            v = float((pred_mwu(m, rval) == yval).mean())
            if v > bv:
                bv, be = v, eta
        m, _ = run_mwu(rtr, ytr, D, be, tro)
        aw.append(metrics(pred_mwu(m, rte), yte)[0])
        mdl, _ = run_signed(rtr, ytr, D, B_bits - 1, LN2, tro)
        a4.append(metrics(pred_signed(mdl, rte), yte)[0])
    A4, AW = np.array(a4), np.array(aw)
    d = A4 - AW
    sd = d.std(ddof=1)
    t = d.mean() / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
    crit = 2.365   # t_{0.05, df=7}
    print("  %-30s 4-bit %.4f+/-%.4f | Warmuth %.4f+/-%.4f | fark %+.2f puan t=%.2f  %s"
          % (name, A4.mean(), A4.std(), AW.mean(), AW.std(), d.mean() * 100, t,
             "BERABERE" if abs(t) < crit else "GERCEK FARK"), flush=True)


print("=" * 96, flush=True)
print("### (a) IKI BASARISIZLIGIN DOGRULANMASI -- 8 tohum, eslesmis t (kritik t=2.365)", flush=True)
print("=" * 96, flush=True)
t0 = time.time()
paired("20NG sci vs talk", lambda: DS.ng(
    ["sci.crypt", "sci.electronics", "sci.med", "sci.space"],
    ["talk.politics.guns", "talk.politics.mideast", "talk.politics.misc",
     "talk.religion.misc"], "x"))
print("  (%.1f dk)" % ((time.time() - t0) / 60), flush=True)
paired("MNIST 0-4 vs 5-9", DS.mnist)
print("  (%.1f dk)" % ((time.time() - t0) / 60), flush=True)

print("\n" + "=" * 96, flush=True)
print("### (b) U-YASASI: GERCEK VERIDE YENIDEN UYDURMA (D-283'un 5 noktasi)", flush=True)
print("=" * 96, flush=True)
U = np.array([2.17, 4.48, 1.18, 4.53, 619.81])
B = np.array([2.0, 3.0, 2.0, 3.0, 5.0])
names = ["20NG comp/rec", "20NG sci/talk", "20NG din/tic", "RCV1", "MNIST"]
x = np.log2(U)
A = np.vstack([x, np.ones(len(x))]).T
slope, icept = np.linalg.lstsq(A, B, rcond=None)[0]
pred = A @ np.array([slope, icept])
r2 = 1 - ((B - pred) ** 2).sum() / ((B - B.mean()) ** 2).sum()
print("  GERCEK VERI uydurmasi : B* = %.3f*log2(U) + %.2f    R^2 = %.3f" % (slope, icept, r2), flush=True)
print("  SENTETIK uydurmasi    : B* = 0.425*log2(U) + 3.29   (D-282)", flush=True)
print("  TEORI                 : egim 0.50", flush=True)
print("\n  %-16s %8s %8s %10s %10s %8s" % ("veri", "U", "olcum", "sentetik", "gercek-fit", "hata"), flush=True)
for i, n in enumerate(names):
    ps = 0.5 * math.log2(U[i]) + 2.6
    print("  %-16s %8.1f %8.0f %10.2f %10.2f %8.2f"
          % (n, U[i], B[i], ps, pred[i], B[i] - ps), flush=True)
errs = np.array([B[i] - (0.5 * math.log2(U[i]) + 2.6) for i in range(5)])
print("\n  Sentetik-kalibreli yasanin hatalari: %s" % np.round(errs, 2).tolist(), flush=True)
print("  >> HEPSI AYNI YONDE (%s) -> sistematik yanlilik, gurultu degil"
      % ("asiri tahmin" if (errs < 0).all() else "eksik tahmin"), flush=True)
print("\ntoplam sure %.1f dk" % ((time.time() - t0) / 60), flush=True)
