# -*- coding: utf-8 -*-
"""DENEY 282: IKI YASAYI DA ATESE AT

Yasa 1 (TEOREM Sonuc 3):  B* >= 1/2 log2(T) + 0.53
   -> iki olcum noktasina dayaniyor. Yasa mi, tesaduf mu?

Yasa 2 (D-278/280 teshisi): belirleyici degisken T degil,
   KOORDINAT BASINA GUNCELLEME  U = M * A / D
   -> hic olculmedi.

TASARIM: B* (gereken en kucuk bit genisligi) iki eksende taranir:
   TEST 1 -- yogunluk SABIT, T degisir     (Yasa 1'i test eder)
   TEST 2 -- T SABIT, yogunluk degisir     (T ve U'yu AYIRIR)

Eger Yasa 1 dogruysa: TEST 1'de B* ~ 1/2 log2 T, TEST 2'de B* SABIT kalmali
   (cunku T sabit).
Eger Yasa 2 dogruysa: TEST 2'de de B* degismeli (cunku U degisiyor),
   ve her iki testte B* ~ 1/2 log2 U tek bir dogruya oturmali.
Bu tasarim ikisini KESIN ayirir.

B* tanimi: cogunluk tabanina gore normalize skor
   z(B) = (acc(B) - cogunluk) / (warmuth - cogunluk)
   B* = z(B) >= 0.90 olan en kucuk B  (yoksa None)
"""
import math, sys, os, time
import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import run_mwu, pred_mwu, to_rows, metrics
from deney_275_tek_ray import run_signed, pred_signed

LN2 = math.log(2.0)
SEEDS = [0, 1, 2]
B_GRID = [2, 3, 4, 5, 6, 7, 8, 10]
ETA_GRID = [0.1, 0.5, LN2, 1.0, 2.0]
D = 10000
K_TRUE = 20
NOISE = 0.05
N_TEST = 2500


def gen(seed, p, n_tr):
    """Dengeli sentetik: 10 arti / 10 eksi isaret, beraberlik yazi-tura."""
    rng = np.random.RandomState(5000 + seed)
    ti = rng.choice(D, K_TRUE, replace=False)
    ts = np.array([1] * (K_TRUE // 2) + [-1] * (K_TRUE // 2))
    rng.shuffle(ts)

    def batch(n):
        rows, ys, done = [], np.zeros(n, np.int8), 0
        while done < n:
            c = min(1000, n - done)
            M = rng.rand(c, D) < p
            s = M[:, ti].astype(np.int8) @ ts
            y = (s > 0).astype(np.int8)
            tie = s == 0
            y[tie] = (rng.rand(int(tie.sum())) < 0.5).astype(np.int8)
            fl = rng.rand(c) < NOISE
            y[fl] = 1 - y[fl]
            ys[done:done + c] = y
            rows.extend(to_rows(sparse.csr_matrix(M)))
            done += c
        return rows, ys

    rtr, ytr = batch(n_tr)
    rte, yte = batch(N_TEST)
    return rtr, ytr, rte, yte


def one_config(p, n_tr):
    """Bir (yogunluk, T) noktasi icin B*, U ve egriyi dondur."""
    accs = {B: [] for B in B_GRID}
    wref, majs, Us, As = [], [], [], []
    for seed in SEEDS:
        rtr, ytr, rte, yte = gen(seed, p, n_tr)
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(rtr))
        nv = int(0.2 * len(order))
        vi, tro = order[:nv], order[nv:]
        rval, yval = [rtr[i] for i in vi], ytr[vi]
        majs.append(max(yte.mean(), 1 - yte.mean()))
        A = float(np.mean([len(r) for r in rtr]))
        As.append(A)

        be, bv = None, -1.0
        for eta in ETA_GRID:
            m, _ = run_mwu(rtr, ytr, D, eta, tro)
            v = float((pred_mwu(m, rval) == yval).mean())
            if v > bv:
                bv, be = v, eta
        m, _ = run_mwu(rtr, ytr, D, be, tro)
        wref.append(metrics(pred_mwu(m, rte), yte)[0])

        for B in B_GRID:
            mdl, mis = run_signed(rtr, ytr, D, B - 1, LN2, tro)   # B bit = isaret + (B-1) us
            accs[B].append(metrics(pred_signed(mdl, rte), yte)[0])
            if B == 4:
                Us.append(mis * A / D)                            # koordinat basina guncelleme

    maj, w = float(np.mean(majs)), float(np.mean(wref))
    span = max(w - maj, 1e-9)
    curve = {B: float(np.mean(accs[B])) for B in B_GRID}
    Bstar = None
    for B in B_GRID:
        if (curve[B] - maj) / span >= 0.90:
            Bstar = B
            break
    return dict(maj=maj, w=w, curve=curve, Bstar=Bstar,
                U=float(np.mean(Us)), A=float(np.mean(As)), T=n_tr)


def report(title, rows, xkey, xlabel):
    print("\n" + "=" * 100, flush=True)
    print("### " + title, flush=True)
    print("=" * 100, flush=True)
    hdr = "%-10s %8s %9s %8s %9s %7s   " % (xlabel, "T", "U=MA/D", "cogunl", "warmuth", "B*")
    hdr += "  ".join("B=%d" % B for B in B_GRID)
    print(hdr, flush=True)
    for r in rows:
        line = "%-10s %8d %9.1f %8.3f %9.3f %7s   " % (
            r[xkey] if xkey != "p" else ("%.3f" % r["p"]),
            r["T"], r["U"], r["maj"], r["w"],
            str(r["Bstar"]) if r["Bstar"] else "->10+")
        line += "  ".join("%.3f" % ((r["curve"][B] - r["maj"]) / max(r["w"] - r["maj"], 1e-9))
                          for B in B_GRID)
        print(line, flush=True)


def fit(rows, key, name):
    xs = [math.log2(r[key]) for r in rows if r["Bstar"]]
    ys = [r["Bstar"] for r in rows if r["Bstar"]]
    if len(xs) < 3:
        print("  %-22s : yeterli nokta yok (%d)" % (name, len(xs)), flush=True)
        return
    A = np.vstack([xs, np.ones(len(xs))]).T
    slope, icept = np.linalg.lstsq(A, np.array(ys, float), rcond=None)[0]
    pred = A @ np.array([slope, icept])
    ss_res = float(((np.array(ys) - pred) ** 2).sum())
    ss_tot = float(((np.array(ys) - np.mean(ys)) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print("  %-22s : B* = %.3f * log2(%s) + %.2f    R^2 = %.3f   (teori egim = 0.50)"
          % (name, slope, key, icept, r2), flush=True)


t0 = time.time()

# --- TEST 1: yogunluk SABIT (p=0.05), T degisiyor -> Yasa 1
rows1 = []
for n_tr in [1000, 2000, 4000, 8000, 16000]:
    r = one_config(0.05, n_tr); r["p"] = 0.05
    rows1.append(r)
    print("  [T=%d] B*=%s  U=%.1f  (%.1f dk)" % (n_tr, r["Bstar"], r["U"], (time.time()-t0)/60), flush=True)
report("TEST 1 -- yogunluk SABIT p=0.05, T degisiyor  (normalize skor z, B* esigi z>=0.90)",
       rows1, "T", "T")

# --- TEST 2: T SABIT (8000), yogunluk degisiyor -> T ve U'yu ayirir
rows2 = []
for p in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
    r = one_config(p, 8000); r["p"] = p
    rows2.append(r)
    print("  [p=%.3f] B*=%s  U=%.1f  A=%.0f  (%.1f dk)" % (p, r["Bstar"], r["U"], r["A"], (time.time()-t0)/60), flush=True)
report("TEST 2 -- T SABIT = 8000, yogunluk degisiyor  (T sabitken B* degisiyor mu?)",
       rows2, "p", "p")

print("\n" + "=" * 100, flush=True)
print("### YASALARIN SINAVI", flush=True)
print("=" * 100, flush=True)
allr = rows1 + rows2
print("\n-- TEST 1 tek basina (yogunluk sabit) --", flush=True)
fit(rows1, "T", "Yasa 1: B* ~ log2 T")
fit(rows1, "U", "Yasa 2: B* ~ log2 U")
print("\n-- TEST 2 tek basina (T sabit) --", flush=True)
fit(rows2, "T", "Yasa 1: B* ~ log2 T")
fit(rows2, "U", "Yasa 2: B* ~ log2 U")
print("\n-- IKI TEST BIRLIKTE (asil sinav: tek dogruya oturuyor mu?) --", flush=True)
fit(allr, "T", "Yasa 1: B* ~ log2 T")
fit(allr, "U", "Yasa 2: B* ~ log2 U")
print("\ntoplam sure %.1f dk" % ((time.time() - t0) / 60), flush=True)
