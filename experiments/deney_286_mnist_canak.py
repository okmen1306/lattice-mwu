# -*- coding: utf-8 -*-
"""DENEY 286: MNIST ACIGINI KAPAT -- YANLIS CANAK HIPOTEZI

DURUM: D-285'te MNIST'te W3'e (MWU+AdaGrad) karsi -1.36 (t=-3.32) geridedeyiz.
Kalan TEK gerilik bu.

HIPOTEZ (teori temelli): MNIST'te YANLIS CANAGI kullaniyoruz.
  MWU'nun gradyan inisini yendigi kosul: KARSILASTIRMACI SEYREK (k << D).
  [Kivinen & Warmuth 1997 -- "EG versus GD" makalesinin tam konusu.]
  MNIST: D=784, A=104 (%13 yogunluk), TUM pikseller sinyal tasiyor
  -> karsilastirmaci YOGUN -> entropi canagi YANLIS UC.
  D-279'da gercek metinde validation r~1-2 (Oklid ucu) secmisti; MNIST test edilmedi.

IKINCI SUPHE: D-284'te MNIST'te secilen k degerleri {0,2,0,1,0} idi ve izgara
  {0,1,2,3} ile sinirliydi. Yogun sentetikte D-281 k=3-4 istemisti. Izgara dar olabilir.

UC DEGISKEN BIRDEN ACILIYOR:
  1) LUT tipi : entropi (2^|s|) vs p-norm (|s|^r), r taranir  -> CANAK
  2) k        : 0..6 (genisletildi)                            -> BIRIKTIRICI
  3) program  : koordinat basina kaydirma (D-284'un kazanani)  -> sabit tutuldu

Baseline: W1 (Warmuth klasik), W2 (kendi programi), W3 (MWU+AdaGrad).
"""
import math, sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import metrics
from deney_285_adil_dovus import mwu_sched, pred_mwu2, ETA_WIDE, C_GRID
import deney_283_coklu_veri as DS

SEEDS = [0, 1, 2]


def make_lut(B, kind, r):
    """Isaretli sayac s in [-S,S] icin LUT. kind='entropi' | 'pnorm'."""
    S = 2 ** (B - 1) - 1
    k = np.arange(-S, S + 1, dtype=np.int64)
    mag = np.abs(k).astype(np.float64)
    if kind == "entropi":
        val = np.where(mag == 0, 0.0, 2.0 ** np.maximum(mag - 1, 0))
    else:
        val = mag ** r
    return np.sign(k) * val, S


def run_engine(rows, ys, D, B, kind, r, k_res, order):
    """Isaretli sayac + k bit artik + koordinat basina kaydirmali program."""
    lut, S = make_lut(B, kind, r)
    s = np.zeros(D, np.int64)
    res = np.zeros(D, np.int64)
    n = np.zeros(D, np.int64)
    mis = 0
    for t in order:
        a = rows[t]
        y = ys[t]
        sc = float(lut[s[a] + S].sum()) if len(a) else 0.0
        pred = 1 if sc > 0 else 0
        if pred != y and len(a):
            mis += 1
            n[a] += 1
            sh = (np.floor(np.log2(np.maximum(n[a], 1))).astype(np.int64)) // 2
            thr = np.left_shift(np.int64(2) ** k_res, sh)
            res[a] += (1 if y == 1 else -1)
            ra = res[a]
            stp = np.where(ra >= thr, 1, np.where(ra <= -thr, -1, 0))
            res[a] = ra - stp * thr
            s[a] = np.clip(s[a] + stp, -S, S)
    return (lut, s, S), mis


def pred_engine(model, rows):
    lut, s, S = model
    out = np.zeros(len(rows), np.int8)
    for i, a in enumerate(rows):
        if len(a):
            out[i] = 1 if float(lut[s[a] + S].sum()) > 0 else 0
    return out


def study(name, loader):
    print("\n" + "=" * 98, flush=True)
    print("### %s" % name, flush=True)
    print("=" * 98, flush=True)
    _, rtr, ytr, rte, yte, D = loader()
    res, cfg = {}, {}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(rtr))
        nv = int(0.2 * len(order))
        vi, tro = order[:nv], order[nv:]
        rval, yval = [rtr[i] for i in vi], ytr[vi]

        # --- Warmuth uc steelman ---
        for lbl, mode, grid in [("W1 Warmuth sabit", "sabit", ETA_WIDE[::2]),
                                ("W2 Warmuth c/sqrt(t)", "kuresel", C_GRID),
                                ("W3 MWU+AdaGrad", "koord", C_GRID)]:
            bb, bv = None, -1.0
            for c in grid:
                m, _ = mwu_sched(rtr, ytr, D, c, mode, tro)
                v = float((pred_mwu2(m, rval) == yval).mean())
                if v > bv:
                    bv, bb = v, c
            m, _ = mwu_sched(rtr, ytr, D, bb, mode, tro)
            res.setdefault(lbl, []).append(metrics(pred_mwu2(m, rte), yte)[0])
            cfg.setdefault(lbl, []).append(bb)

        # --- ENTROPI canagi, genisletilmis k ---
        bb, bv = None, -1.0
        for kk in [0, 1, 2, 3, 4, 6]:
            m, _ = run_engine(rtr, ytr, D, 4, "entropi", 0, kk, tro)
            v = float((pred_engine(m, rval) == yval).mean())
            if v > bv:
                bv, bb = v, kk
        m, _ = run_engine(rtr, ytr, D, 4, "entropi", 0, bb, tro)
        res.setdefault("BIZ entropi (4+k bit)", []).append(metrics(pred_engine(m, rte), yte)[0])
        cfg.setdefault("BIZ entropi (4+k bit)", []).append("k=%d" % bb)

        # --- p-NORM canagi: r ve k birlikte ---
        bb, bv = None, -1.0
        for rr in [1.0, 2.0, 4.0]:
            for kk in [0, 3]:
                m, _ = run_engine(rtr, ytr, D, 6, "pnorm", rr, kk, tro)
                v = float((pred_engine(m, rval) == yval).mean())
                if v > bv:
                    bv, bb = v, (rr, kk)
        m, _ = run_engine(rtr, ytr, D, 6, "pnorm", bb[0], bb[1], tro)
        res.setdefault("BIZ p-norm (6+k bit)", []).append(metrics(pred_engine(m, rte), yte)[0])
        cfg.setdefault("BIZ p-norm (6+k bit)", []).append("r=%g,k=%d" % bb)
        print("  seed %d bitti" % seed, flush=True)

    order_lbl = ["W1 Warmuth sabit", "W2 Warmuth c/sqrt(t)", "W3 MWU+AdaGrad",
                 "BIZ entropi (4+k bit)", "BIZ p-norm (6+k bit)"]
    best_ours = max(["BIZ entropi (4+k bit)", "BIZ p-norm (6+k bit)"],
                    key=lambda L: np.mean(res[L]))
    ours = np.array(res[best_ours])
    print("\n  %-24s %9s %9s   %s" % ("model", "acc", "std", "en iyi bize gore"), flush=True)
    print("  " + "-" * 88, flush=True)
    for lbl in order_lbl:
        a = np.array(res[lbl])
        tag = ""
        if not lbl.startswith("BIZ"):
            d = ours - a
            sd = d.std(ddof=1)
            t = d.mean() / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
            crit = 4.303   # df=2
            tag = "biz-o: %+6.2f (t=%6.2f)%s" % (
                d.mean() * 100, t,
                "  BIZ KAZANDIK" if t > crit else ("  BIZ KAYBETTIK" if t < -crit else "  BERABERE"))
        star = " <== en iyi bizim" if lbl == best_ours else ""
        print("  %-24s %9.4f %9.4f   %s%s" % (lbl, a.mean(), a.std(), tag, star), flush=True)
    print("  secilenler: %s" % {k: v for k, v in cfg.items() if k.startswith("BIZ")}, flush=True)


if __name__ == "__main__":
    t0 = time.time()
    study("MNIST 0-4 vs 5-9  (kalan TEK gerilik)", DS.mnist)
    print("  (%.1f dk)" % ((time.time() - t0) / 60), flush=True)
    study("KONTROL: 20NG comp vs rec (seyrek -- entropi burada dogru canak olmali)",
          DS.LOADERS[0])
    print("\ntoplam sure %.1f dk" % ((time.time() - t0) / 60), flush=True)
