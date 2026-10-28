# -*- coding: utf-8 -*-
"""DENEY 275: TEK RAY + ISARET BITI ("kaydirmali ustel isaret") vs CIFT RAY.

Soru: cift rayli (c_pos, c_neg) yapiyi birakip tek ray + 1 isaret bitine
gecersek, ESIT BELLEKTE Warmuth'u yakalar/gecer miyiz? 3 bit yeter mi? 4 bit?

Temsil:  phi(s,e) = 0            eger e = 0   (olu sinaps)
                  = s * 2^(e-1)  eger e >= 1  (saf bit-shift, s in {+1,-1})
Bellek : 1 isaret biti + B_exp bit = B_exp + 1 bit / sinaps
         (cift rayda 2*B bit / sinaps idi)

Guncelleme (hata aninda, yon d = +1 eger y==1 degilse -1):
    e == 0            -> e = 1,  s = d          (olu sinaps yon secer)
    e > 0 ve s == d   -> e = min(EMAX, e+1)     (guveni 2 katla)
    e > 0 ve s != d   -> e = e - 1              (guveni yarila; 0'da olur)
Sifir gecisi bu sekilde dogal: +1 -> 0 -> -1.
"""
import math, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import (run_mwu, pred_mwu, run_quant, pred_quant,
                  make_synth, load_real, LN2, ETA_GRID, metrics, mcnemar)

SEEDS = [0, 1, 2, 3, 4]


# ------------------------------------------------- TEK RAY + ISARET
def run_signed(rows, ys, D, B_exp, eta, order):
    EMAX = 2 ** B_exp - 1
    lut = np.array([0.0] + [math.exp(i * eta) for i in range(EMAX)])
    e = np.zeros(D, np.int16)
    s = np.ones(D, np.int8)
    mis = 0
    for t in order:
        a = rows[t]
        y = ys[t]
        if len(a):
            ea = e[a]
            pred = 1 if float((s[a] * lut[ea]).sum()) > 0 else 0
        else:
            pred = 0
        if pred != y:
            mis += 1
            if len(a):
                d = np.int8(1) if y == 1 else np.int8(-1)
                ea = e[a]
                sa = s[a]
                zero = ea == 0
                same = (~zero) & (sa == d)
                opp = (~zero) & (sa != d)
                ea = np.where(same, np.minimum(EMAX, ea + 1), ea)
                ea = np.where(opp, ea - 1, ea)
                ea = np.where(zero, 1, ea)
                sa = np.where(zero, d, sa).astype(np.int8)
                e[a] = ea
                s[a] = sa
    return ("signed", lut, e, s), mis


def pred_signed(model, rows):
    _, lut, e, s = model
    out = np.zeros(len(rows), np.int8)
    for i, a in enumerate(rows):
        if len(a):
            out[i] = 1 if float((s[a] * lut[e[a]]).sum()) > 0 else 0
    return out


# ------------------------------------------------- deney
def phase_A(name, get_data):
    """eta=ln2 SABIT (saf 1<<k). Bellek butcesine gore iki aile karsi karsiya."""
    print("\n" + "=" * 84)
    print("### %s" % name)
    print("### FAZ A -- eta=ln2 SABIT (saf bit-shift), bellek butcesine gore")
    print("=" * 84)
    acc = {}
    for seed in SEEDS:
        rtr, ytr, rte, yte, D = get_data(seed)
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(rtr))
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
        acc.setdefault(("Warmuth MWU (eta* ayarli, FP64)", 64), []).append(
            metrics(pred_mwu(m, rte), yte)[0])

        for Be in [2, 3, 4, 5, 6]:                      # tek ray: Be+1 bit
            m, _ = run_signed(rtr, ytr, D, Be, LN2, tro)
            acc.setdefault(("TEK RAY  isaret+%d us" % Be, Be + 1), []).append(
                metrics(pred_signed(m, rte), yte)[0])
        for B in [2, 3, 4, 6]:                          # cift ray: 2B bit
            m, _ = run_quant(rtr, ytr, D, B, LN2, tro)
            acc.setdefault(("CIFT RAY %d+%d" % (B, B), 2 * B), []).append(
                metrics(pred_quant(m, rte), yte)[0])

    ref = np.array(acc[("Warmuth MWU (eta* ayarli, FP64)", 64)])
    print("%-26s %6s %9s %8s   %s" % ("model", "bit", "acc%", "std", "Warmuth'a fark"))
    for k in sorted(acc, key=lambda z: (z[1], z[0])):
        a = np.array(acc[k])
        tag = "referans"
        if k[1] != 64:
            d = a - ref
            sd = d.std(ddof=1)
            t = d.mean() / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
            mark = "  <-- BERABERE" if abs(t) < 2.776 else ""
            tag = "%+6.2f puan  t=%6.2f%s" % (d.mean() * 100, t, mark)
        print("%-26s %6s %9.2f %8.2f   %s"
              % (k[0], k[1] if k[1] != 64 else "fp64", a.mean() * 100, a.std() * 100, tag))
    return acc


def phase_B(name, get_data, widths):
    """3 ve 4 bit TEK RAY: eta ayarlanirsa kurtulur mu? (bit-shift'ten feragat)"""
    print("\n" + "-" * 84)
    print("### FAZ B -- %s : dar tek ray, eta validation'da ayarlaniyor" % name)
    print("-" * 84)
    acc, chosen = {}, {}
    for seed in SEEDS:
        rtr, ytr, rte, yte, D = get_data(seed)
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(rtr))
        nv = int(0.2 * len(order))
        vi, tro = order[:nv], order[nv:]
        rval, yval = [rtr[i] for i in vi], ytr[vi]
        for Be in widths:
            best, bv = None, -1.0
            for eta in ETA_GRID:
                m, _ = run_signed(rtr, ytr, D, Be, eta, tro)
                v = float((pred_signed(m, rval) == yval).mean())
                if v > bv:
                    bv, best = v, eta
            m, _ = run_signed(rtr, ytr, D, Be, best, tro)
            acc.setdefault(Be, []).append(metrics(pred_signed(m, rte), yte)[0])
            chosen.setdefault(Be, []).append(best)
    for Be in widths:
        a = np.array(acc[Be])
        print("TEK RAY isaret+%d us (%d bit, eta ayarli): %.2f%% +/-%.2f   secilen eta: %s"
              % (Be, Be + 1, a.mean() * 100, a.std() * 100,
                 ["%.3g" % x for x in chosen[Be]]))


if __name__ == "__main__":
    phase_A("SENTETIK DENGELI (yogun: ~1000 aktif/ornek)", make_synth)
    phase_B("SENTETIK DENGELI", make_synth, [2, 3])
    phase_A("GERCEK VERI -- 20 Newsgroups (seyrek: ~44 aktif/dok)", load_real)
    phase_B("GERCEK VERI", load_real, [2, 3])
