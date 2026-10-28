# -*- coding: utf-8 -*-
"""DENEY 278: FLOAT ILE OGRENMEK vs TAMSAYI ILE OGRENMEK

KAMA: BitNet-tarzi QAT, float master agirlik tutar ve ileri geciste kuantize eder.
Bizim motorumuz float master TUTMAZ; durum dogrudan tamsayidir.

Bu deney farki TAM olarak izole eder:
  - Ayni temsil (tek ray + isaret), ayni kafes (delta=ln2), ayni kutu (B bit),
    ayni CIKARIM maliyeti. TEK FARK: dual degiskenin float golge kopyasi.
  - Float master'in tek gercek avantaji KUANTUM ALTI adimlari kayipsiz
    biriktirebilmesidir (ornt. adim +0.05 tam olarak islenir).
    Tamsayi motor bunu ancak STOKASTIK olarak (kayipli) yapabilir (D-276).

HIPOTEZ: eger ikisi beraberse float master gereksizdir -> egitim durumu
64 bit yerine B bit (16x). Eger master ezerse, kamamiz coker.

Kollar (her ikisi de ayni B bitlik cikarim yapar):
  A) Warmuth FP64            -- hic kuantizasyon yok, ust referans (64 bit)
  B) MASTER-FLOAT (QAT)      -- float ustel + yuvarlanmis ileri gecis (64+1 bit)
  C) TAMSAYI-SAF (bizimki)   -- durum dogrudan tamsayi, stokastik adim (B bit)
Her kolun adim buyuklugu p validation'da ayarlanir.
"""
import math, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import run_mwu, pred_mwu, make_synth, load_real, LN2, ETA_GRID, metrics
from deney_275_tek_ray import pred_signed
from deney_276_stokastik_kafes import run_signed_p

SEEDS = [0, 1, 2, 3, 4]
P_GRID = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
WIDTHS = [3, 5]          # B_exp -> 4 bit ve 6 bit toplam


# ------------------------------------------------ B) MASTER-FLOAT (QAT)
def run_master(rows, ys, D, B_exp, p, order):
    """Float master ustel + kuantize ileri gecis (BitNet-tarzi QAT analogu).
    Adim p KESIN olarak islenir (kuantum alti bilgi kaybolmaz)."""
    EMAX = 2 ** B_exp - 1
    lut = np.array([0.0] + [2.0 ** i for i in range(EMAX)])
    ef = np.zeros(D)                  # FLOAT master ustel
    s = np.ones(D, np.int8)
    mis = 0
    for t in order:
        a = rows[t]
        y = ys[t]
        pred = 0
        if len(a):
            eq = np.clip(np.rint(ef[a]), 0, EMAX).astype(np.int64)   # ileri geciste kuantize
            pred = 1 if float((s[a] * lut[eq]).sum()) > 0 else 0
        if pred != y:
            mis += 1
            if len(a):
                d = np.int8(1) if y == 1 else np.int8(-1)
                ea, sa = ef[a], s[a]
                same = sa == d
                ea = np.where(same, np.minimum(EMAX, ea + p), ea - p)
                flip = ea < 0                       # sifir gecisi: yansit, isaret degistir
                sa = np.where(flip, d, sa).astype(np.int8)
                ea = np.where(flip, -ea, ea)
                ef[a] = np.minimum(EMAX, ea)
                s[a] = sa
    eq = np.clip(np.rint(ef), 0, EMAX).astype(np.int64)
    return ("signed", lut, eq, s), mis


def tune_and_eval(runner, rtr, ytr, rte, rval, yval, D, B_exp, tro, seed, needs_rng):
    best, bv = None, -1.0
    for p in P_GRID:
        if needs_rng:
            g = np.random.default_rng(40000 + seed)
            m, _ = runner(rtr, ytr, D, B_exp, p, tro, g)
        else:
            m, _ = runner(rtr, ytr, D, B_exp, p, tro)
        v = float((pred_signed(m, rval) == yval).mean())
        if v > bv:
            bv, best = v, p
    if needs_rng:
        g = np.random.default_rng(40000 + seed)
        m, mis = runner(rtr, ytr, D, B_exp, best, tro, g)
    else:
        m, mis = runner(rtr, ytr, D, B_exp, best, tro)
    return pred_signed(m, rte), best, mis


def study(name, get_data):
    print("\n" + "=" * 90, flush=True)
    print("### %s" % name, flush=True)
    print("=" * 90, flush=True)
    ref, res, chosen = [], {}, {}
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

        for Be in WIDTHS:
            pm, bp, _ = tune_and_eval(run_master, rtr, ytr, rte, rval, yval, D, Be, tro, seed, False)
            res.setdefault(("B) MASTER-FLOAT", Be), []).append(metrics(pm, yte)[0])
            chosen.setdefault(("B) MASTER-FLOAT", Be), []).append(bp)

            pi, bp, _ = tune_and_eval(run_signed_p, rtr, ytr, rte, rval, yval, D, Be, tro, seed, True)
            res.setdefault(("C) TAMSAYI-SAF", Be), []).append(metrics(pi, yte)[0])
            chosen.setdefault(("C) TAMSAYI-SAF", Be), []).append(bp)
        print("  seed %d bitti" % seed, flush=True)

    R = np.array(ref)
    print("\n  A) Warmuth FP64 (eta* ayarli, 64 bit/sinaps): %.2f%% +/-%.2f\n"
          % (R.mean() * 100, R.std() * 100), flush=True)
    print("  %-18s %6s %10s %9s %14s   %s"
          % ("kol", "cikarim", "egitim", "acc%", "std", "secilen p"), flush=True)
    print("  " + "-" * 84, flush=True)
    for Be in WIDTHS:
        bits = Be + 1
        for arm in ["B) MASTER-FLOAT", "C) TAMSAYI-SAF"]:
            a = np.array(res[(arm, Be)])
            train_bits = "64+1 bit" if arm.startswith("B") else "%d bit" % bits
            print("  %-18s %5d b %10s %9.2f %14.2f   %s"
                  % (arm, bits, train_bits, a.mean() * 100, a.std() * 100,
                     ["%g" % x for x in chosen[(arm, Be)]]), flush=True)
        am = np.array(res[("B) MASTER-FLOAT", Be)])
        ai = np.array(res[("C) TAMSAYI-SAF", Be)])
        d = ai - am
        sd = d.std(ddof=1)
        t = d.mean() / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
        mark = "BERABERE -> float master GEREKSIZ" if abs(t) < 2.776 else "MASTER KAZANIYOR -> kama zayif"
        print("     >> %d bit: tamsayi - master = %+.2f puan (t=%.2f)  %s\n"
              % (bits, d.mean() * 100, t, mark), flush=True)


study("GERCEK VERI -- 20 Newsgroups (seyrek)", load_real)
study("SENTETIK DENGELI (yogun)", make_synth)
