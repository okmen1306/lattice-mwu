# -*- coding: utf-8 -*-
"""DENEY 276: STOKASTIK KAFES ADIMI (dithered lattice MWU).

TEORI: kafes aralIgI delta=ln2 SABIT kalir (saf bit-shift korunur), ama adim
olasiliksal atilir: her aktif sayac hata aninda p olasilikla +-1 kayar.
   E[adim] = p*delta  ->  EFEKTIF ogrenme orani eta = p*ln2 < ln2
Yani kaba kafeste ince ogrenme orani. Bellek ayni, `1<<k` ozelligi ayni.

YANLISLANABILIR TAHMIN (rastgele yuruyus argumani):
   M hata sonrasi sayac salinimi ~ delta*sqrt(p*M);
   kutuya carpmamasi icin sqrt(p*M) <~ N  =>  p* ~ N^2 / M
   M~4500 icin:  B=6 (N=63) -> p*~0.88   B=5 (N=31) -> p*~0.21   B=4 (N=15) -> p*~0.05
Eger olculen optimum p bu degerlere yakin cikarsa teori dogrulanir.
"""
import math, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import (run_mwu, pred_mwu, pred_quant, make_synth, load_real,
                  LN2, ETA_GRID, metrics)
from deney_275_tek_ray import pred_signed

SEEDS = [0, 1, 2, 3, 4]
P_GRID = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]


def run_quant_p(rows, ys, D, B, p, order, rng):
    """Cift ray, eta=ln2 SABIT, adim olasiligi p (p=1 -> D273/D274 motoru)."""
    S = 2 ** B - 1
    lut = np.array([0.0] + [2.0 ** i for i in range(S)])
    cp = np.ones(D, np.int16)
    cn = np.ones(D, np.int16)
    mis = 0
    for t in order:
        a = rows[t]
        y = ys[t]
        pred = 0
        if len(a):
            pred = 1 if (lut[cp[a]].sum() - lut[cn[a]].sum()) > 0 else 0
        if pred != y:
            mis += 1
            if len(a):
                m = rng.random(len(a)) < p if p < 1.0 else np.ones(len(a), bool)
                aa = a[m]
                if len(aa):
                    if y == 1:
                        cp[aa] = np.minimum(S, cp[aa] + 1)
                        cn[aa] = np.maximum(0, cn[aa] - 1)
                    else:
                        cp[aa] = np.maximum(0, cp[aa] - 1)
                        cn[aa] = np.minimum(S, cn[aa] + 1)
    return ("quant", lut, cp, cn), mis


def run_signed_p(rows, ys, D, B_exp, p, order, rng):
    """Tek ray + isaret, eta=ln2 SABIT, adim olasiligi p."""
    EMAX = 2 ** B_exp - 1
    lut = np.array([0.0] + [2.0 ** i for i in range(EMAX)])
    e = np.zeros(D, np.int16)
    s = np.ones(D, np.int8)
    mis = 0
    for t in order:
        a = rows[t]
        y = ys[t]
        pred = 0
        if len(a):
            pred = 1 if float((s[a] * lut[e[a]]).sum()) > 0 else 0
        if pred != y:
            mis += 1
            if len(a):
                m = rng.random(len(a)) < p if p < 1.0 else np.ones(len(a), bool)
                aa = a[m]
                if len(aa):
                    d = np.int8(1) if y == 1 else np.int8(-1)
                    ea, sa = e[aa], s[aa]
                    zero = ea == 0
                    same = (~zero) & (sa == d)
                    opp = (~zero) & (sa != d)
                    ea = np.where(same, np.minimum(EMAX, ea + 1), ea)
                    ea = np.where(opp, ea - 1, ea)
                    ea = np.where(zero, 1, ea)
                    e[aa] = ea
                    s[aa] = np.where(zero, d, sa).astype(np.int8)
    return ("signed", lut, e, s), mis


def study(name, get_data, dual_widths, signed_widths):
    print("\n" + "=" * 88, flush=True)
    print("### %s" % name, flush=True)
    print("=" * 88, flush=True)
    ref, res, mism = [], {}, {}
    for seed in SEEDS:
        rtr, ytr, rte, yte, D = get_data(seed)
        r = np.random.RandomState(seed)
        order = r.permutation(len(rtr))
        nv = int(0.2 * len(order))
        vi, tro = order[:nv], order[nv:]
        rval, yval = [rtr[i] for i in vi], ytr[vi]

        be, bv = None, -1.0
        for eta in ETA_GRID:
            m, _ = run_mwu(rtr, ytr, D, eta, tro)
            v = float((pred_mwu(m, rval) == yval).mean())
            if v > bv:
                bv, be = v, eta
        m, _ = run_mwu(rtr, ytr, D, be, tro)
        ref.append(metrics(pred_mwu(m, rte), yte)[0])

        for B in dual_widths:
            for p in P_GRID:
                g = np.random.default_rng(10000 + seed)
                m, mi = run_quant_p(rtr, ytr, D, B, p, tro, g)
                res.setdefault(("CIFT %d+%d (%2d bit)" % (B, B, 2 * B), p), []).append(
                    metrics(pred_quant(m, rte), yte)[0])
                mism.setdefault(("CIFT %d+%d (%2d bit)" % (B, B, 2 * B), p), []).append(mi)
        for Be in signed_widths:
            for p in P_GRID:
                g = np.random.default_rng(20000 + seed)
                m, mi = run_signed_p(rtr, ytr, D, Be, p, tro, g)
                res.setdefault(("TEK  isaret+%d (%2d bit)" % (Be, Be + 1), p), []).append(
                    metrics(pred_signed(m, rte), yte)[0])
                mism.setdefault(("TEK  isaret+%d (%2d bit)" % (Be, Be + 1), p), []).append(mi)
        print("  seed %d bitti" % seed, flush=True)

    R = np.array(ref)
    print("\nWarmuth MWU (eta* ayarli, FP64) referans: %.2f%% +/-%.2f\n"
          % (R.mean() * 100, R.std() * 100), flush=True)
    header = "%-24s" % "model" + "".join("%9s" % ("p=%g" % p) for p in P_GRID) + "   en iyi p"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    models = sorted({k[0] for k in res})
    for mdl in models:
        row = "%-24s" % mdl
        best, bp = -1, None
        for p in P_GRID:
            a = np.array(res[(mdl, p)]).mean() * 100
            row += "%9.2f" % a
            if a > best:
                best, bp = a, p
        M = np.array(mism[(mdl, bp)]).mean()
        N = int(mdl.split("+")[1].split(" ")[0]) if "isaret" in mdl else int(mdl.split()[1].split("+")[0])
        N = 2 ** N - 1
        row += "   p*=%-5g (tahmin %.3f, M=%.0f)" % (bp, min(1.0, N * N / max(M, 1)), M)
        print(row, flush=True)


if __name__ == "__main__":
    study("SENTETIK DENGELI (yogun ~1000 aktif/ornek) -- ASIL SINAV",
          make_synth, [4, 5, 6], [3, 5])
    study("GERCEK VERI 20 Newsgroups (seyrek ~44 aktif/dok)",
          load_real, [4], [3])
