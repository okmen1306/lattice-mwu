# -*- coding: utf-8 -*-
"""DENEY 284: AZALAN ADIM PROGRAMI (kafeste, kaydirmayla)

BOSLUK: butun motorlarimiz SABIT efektif adim kullandi (p bir kez ayarlandi).
Teorinin sifir-pismanlik verdigi sey ise AZALAN adimdir (eta_t ~ 1/sqrt(t)).
Bunu hic test etmedik.

Kafeste bedava oturuyor: artik biriktiricinin ESIGINI kaydirmak yeterli.
   p = g / 2^(k + shift),   shift = floor(log2(sayac)) // 2
   -> sayac 4 katina cikinca esik 1 bit kayar, adim yariya iner
   -> tam olarak p ~ 1/sqrt(sayac),  SIFIR bolme, SIFIR float

UC KOL:
  A) SABIT      : shift = 0            (D-281 motoru, baseline)
  B) KURESEL    : shift = f(t)         (Warmuth'un klasik 1/sqrt(t) programi)
  C) KOORDINAT  : shift = f(n_i)       (D-282: belirleyici degisken U = koordinat
                                        basina guncelleme -> program da oyle olmali)

En kritik tezgah MNIST: kaybettigimiz TEK yer (-3.96, t=-2.52) ve U=620 ile
en yuksek guncelleme yogunluguna sahip olan.
"""
import math, sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import run_mwu, pred_mwu, make_synth, metrics
from deney_275_tek_ray import pred_signed
import deney_283_coklu_veri as DS

LN2 = math.log(2.0)
SEEDS = [0, 1, 2, 3, 4]
B_EXP = 3                      # 4 bit isaretli sayac
MODES = ["A-sabit", "B-kuresel", "C-koordinat"]


def run_sched(rows, ys, D, k, mode, order):
    """4 bit sayac + k bit artik; esik kaydirmali azalan adim."""
    EMAX = 2 ** B_EXP - 1
    lut = np.array([0.0] + [2.0 ** i for i in range(EMAX)])
    e = np.zeros(D, np.int16)
    s = np.ones(D, np.int8)
    r = np.zeros(D, np.int64)
    n = np.zeros(D, np.int64)          # koordinat basina guncelleme sayaci
    mis = 0
    for step, t in enumerate(order, start=1):
        a = rows[t]
        y = ys[t]
        sc = float((s[a] * lut[e[a]]).sum()) if len(a) else 0.0
        pred = 1 if sc > 0 else 0
        if pred != y:
            mis += 1
            if len(a):
                if mode == "A-sabit":
                    sh = np.zeros(len(a), np.int64)
                elif mode == "B-kuresel":
                    sh = np.full(len(a), (max(mis, 1).bit_length() - 1) // 2, np.int64)
                else:  # C-koordinat
                    n[a] += 1
                    sh = (np.floor(np.log2(np.maximum(n[a], 1))).astype(np.int64)) // 2
                thr = np.left_shift(np.int64(2) ** k, sh)      # 2^(k+shift)
                d = 1 if y == 1 else -1
                r[a] += d
                ra = r[a]
                stp = np.where(ra >= thr, 1, np.where(ra <= -thr, -1, 0)).astype(np.int8)
                r[a] = ra - stp.astype(np.int64) * thr
                m = stp != 0
                if m.any():
                    aa, dd = a[m], stp[m]
                    ea, sa = e[aa], s[aa]
                    zero = ea == 0
                    same = (~zero) & (sa == dd)
                    opp = (~zero) & (sa != dd)
                    ea = np.where(same, np.minimum(EMAX, ea + 1), ea)
                    ea = np.where(opp, ea - 1, ea)
                    ea = np.where(zero, 1, ea)
                    e[aa] = ea
                    s[aa] = np.where(zero, dd, sa).astype(np.int8)
    return ("signed", lut, e, s), mis


def study(name, get_data):
    print("\n" + "=" * 94, flush=True)
    print("### %s" % name, flush=True)
    print("=" * 94, flush=True)
    ref, res, ks = [], {}, {}
    for seed in SEEDS:
        rtr, ytr, rte, yte, D = get_data(seed)
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
        ref.append(metrics(pred_mwu(m, rte), yte)[0])

        for mode in MODES:
            bk, bkv = None, -1.0
            for k in [0, 1, 2, 3]:
                mdl, _ = run_sched(rtr, ytr, D, k, mode, tro)
                v = float((pred_signed(mdl, rval) == yval).mean())
                if v > bkv:
                    bkv, bk = v, k
            mdl, _ = run_sched(rtr, ytr, D, bk, mode, tro)
            res.setdefault(mode, []).append(metrics(pred_signed(mdl, rte), yte)[0])
            ks.setdefault(mode, []).append(bk)
        print("  seed %d bitti" % seed, flush=True)

    R = np.array(ref)
    print("\n  Warmuth FP64 (64 bit): %.4f +/-%.4f\n" % (R.mean(), R.std()), flush=True)
    base = np.array(res["A-sabit"])
    for mode in MODES:
        a = np.array(res[mode])
        dW = a - R
        sdW = dW.std(ddof=1)
        tW = dW.mean() / (sdW / math.sqrt(len(dW))) if sdW > 0 else float("nan")
        line = "  %-13s %.4f +/-%.4f | Warmuth'a %+6.2f (t=%6.2f)%s" % (
            mode, a.mean(), a.std(), dW.mean() * 100, tW,
            "  BERABERE" if abs(tW) < 2.776 else "")
        if mode != "A-sabit":
            d = a - base
            sd = d.std(ddof=1)
            t = d.mean() / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
            line += " | sabite %+6.2f (t=%6.2f)%s" % (
                d.mean() * 100, t, "  KAZANDI" if t > 2.776 else "")
        line += "  k: %s" % ks[mode]
        print(line, flush=True)


if __name__ == "__main__":
    t0 = time.time()
    study("MNIST 0-4 vs 5-9  (U=620, KAYBETTIGIMIZ TEK YER)",
          lambda s: DS.mnist()[1:])
    print("  (%.1f dk)" % ((time.time() - t0) / 60), flush=True)
    study("YOGUN SENTETIK (U~450)", make_synth)
    print("  (%.1f dk)" % ((time.time() - t0) / 60), flush=True)
    study("20NG comp vs rec (U=2.2, zaten berabere)",
          lambda s: DS.LOADERS[0]()[1:])
    print("\ntoplam sure %.1f dk" % ((time.time() - t0) / 60), flush=True)
