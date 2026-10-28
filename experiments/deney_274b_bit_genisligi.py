# -*- coding: utf-8 -*-
"""D274-B: SAF BIT-SHIFT (eta=ln2 SABIT) bit-genisligi taramasi.

Soru: D273 motorunun cokusu (a) bit-shift fikrinden mi, (b) yoksa sadece
B=4'un dar dinamik araligindan mi kaynaklaniyor?
eta=ln2 SABIT tutulur -> LUT = [0,1,2,4,8,...] yani saf `1<<k`, sifir float carpma.
Sadece B degistirilir. Kiyas: ayarlanmis FP64 Warmuth.
"""
import math, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import (run_mwu, pred_mwu, run_quant, pred_quant, make_synth,
                  load_real, LN2, ETA_GRID, metrics)

SEEDS = [0, 1, 2, 3, 4]
B_LIST = [3, 4, 5, 6, 7, 8, 10]


def sweep(name, get_data):
    print("\n" + "=" * 76)
    print("### %s  --  saf bit-shift (eta=ln2), sadece B degisiyor" % name)
    print("=" * 76)
    acc = {}
    for seed in SEEDS:
        rtr, ytr, rte, yte, D = get_data(seed)
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(rtr))
        nv = int(0.2 * len(order))
        vi, tro = order[:nv], order[nv:]
        rval, yval = [rtr[i] for i in vi], ytr[vi]

        be, bv = None, -1.0
        for eta in ETA_GRID:                      # Warmuth val'de ayarlanir
            m, _ = run_mwu(rtr, ytr, D, eta, tro)
            v = float((pred_mwu(m, rval) == yval).mean())
            if v > bv:
                bv, be = v, eta
        m, _ = run_mwu(rtr, ytr, D, be, tro)
        acc.setdefault("Warmuth MWU (eta* ayarli, FP64)", []).append(
            metrics(pred_mwu(m, rte), yte)[0])

        for B in B_LIST:
            m, _ = run_quant(rtr, ytr, D, B, LN2, tro)
            acc.setdefault("BitShift B=%-2d (saf 1<<k)" % B, []).append(
                metrics(pred_quant(m, rte), yte)[0])

    print("%-36s %10s %10s" % ("model", "acc%", "std"))
    ref = np.array(acc["Warmuth MWU (eta* ayarli, FP64)"])
    for k in list(acc):
        a = np.array(acc[k])
        tag = ""
        if k.startswith("BitShift"):
            d = a - ref
            sd = d.std(ddof=1)
            t = d.mean() / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
            tag = "  (Warmuth'a fark %+.2f, t=%.2f)" % (d.mean() * 100, t)
        print("%-36s %10.2f %10.2f%s" % (k, a.mean() * 100, a.std() * 100, tag))


sweep("SENTETIK DENGELI", make_synth)
sweep("GERCEK VERI (20 Newsgroups)", load_real)
