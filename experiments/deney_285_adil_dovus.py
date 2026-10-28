# -*- coding: utf-8 -*-
"""DENEY 285: ADIL DOVUS -- D-284'un "zaferi" baseline artefakti mi?

D-284'te C-koordinat MNIST'te Warmuth'u +4.21 (t=6.32), B-kuresel yogunda
+4.07 (t=9.06) gecti. IKI KIRMIZI BAYRAK:

  1) Warmuth'a verilen eta izgarasi [0.1 .. 2.0] -- EN KUCUGU 0.1.
     Bizim motorun efektif adimi p=1/2^k; k=3 ile eta_eff=ln2/8=0.087,
     yani ZATEN IZGARANIN ALTINDA. Warmuth erisemedigi bir bolgede yaris.
  2) Bizimkinin AZALAN programi var, Warmuth'unki SABIT. Elmayla armut.

Bu deney Warmuth'a UC steelman verir:
  W1) sabit eta, GENISLETILMIS izgara (0.005 ... 4)
  W2) kuresel azalan program eta_t = c/sqrt(t)
  W3) KOORDINAT BASINA azalan program eta_i = c/sqrt(n_i)  <-- birebir eslenik

Eger C-koordinat W3'u de geciyorsa sonuc gercektir. Gecemiyorsa
D-284'un zaferi ARTEFAKTTIR ve oyle yazilir.
"""
import math, sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import make_synth, metrics
from deney_275_tek_ray import pred_signed
from deney_284_azalan_program import run_sched
import deney_283_coklu_veri as DS

LN2 = math.log(2.0)
SEEDS = [0, 1, 2, 3, 4]
ETA_WIDE = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, LN2, 1.0, 2.0, 4.0]
C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def mwu_sched(rows, ys, D, c, mode, order):
    """Log-uzayi MWU; mode: 'sabit' (c=eta), 'kuresel' (c/sqrt(t)), 'koord' (c/sqrt(n_i))."""
    lp = np.zeros(D); lnn = np.zeros(D); n = np.zeros(D, np.int64); mis = 0
    for t in order:
        a = rows[t]; y = ys[t]
        pred = 0
        if len(a):
            u, v = lp[a], lnn[a]
            m = max(u.max(), v.max())
            pred = 1 if (np.exp(u - m).sum() - np.exp(v - m).sum()) > 0 else 0
        if pred != y:
            mis += 1
            if len(a):
                if mode == "sabit":
                    eta = c
                elif mode == "kuresel":
                    eta = c / math.sqrt(max(mis, 1))
                else:
                    n[a] += 1
                    eta = c / np.sqrt(np.maximum(n[a], 1))
                d = 1.0 if y == 1 else -1.0
                lp[a] += d * eta
                lnn[a] -= d * eta
    return ("mwu", lp, lnn), mis


def pred_mwu2(model, rows):
    _, lp, lnn = model
    out = np.zeros(len(rows), np.int8)
    for i, a in enumerate(rows):
        if len(a):
            u, v = lp[a], lnn[a]
            m = max(u.max(), v.max())
            out[i] = 1 if (np.exp(u - m).sum() - np.exp(v - m).sum()) > 0 else 0
    return out


def study(name, get_data):
    print("\n" + "=" * 96, flush=True)
    print("### %s" % name, flush=True)
    print("=" * 96, flush=True)
    res, cfg = {}, {}
    for seed in SEEDS:
        rtr, ytr, rte, yte, D = get_data(seed)
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(rtr))
        nv = int(0.2 * len(order))
        vi, tro = order[:nv], order[nv:]
        rval, yval = [rtr[i] for i in vi], ytr[vi]

        for lbl, mode, grid in [("W1 sabit (genis izgara)", "sabit", ETA_WIDE),
                                ("W2 kuresel c/sqrt(t)", "kuresel", C_GRID),
                                ("W3 koordinat c/sqrt(n_i)", "koord", C_GRID)]:
            bb, bv = None, -1.0
            for c in grid:
                m, _ = mwu_sched(rtr, ytr, D, c, mode, tro)
                v = float((pred_mwu2(m, rval) == yval).mean())
                if v > bv:
                    bv, bb = v, c
            m, _ = mwu_sched(rtr, ytr, D, bb, mode, tro)
            res.setdefault(lbl, []).append(metrics(pred_mwu2(m, rte), yte)[0])
            cfg.setdefault(lbl, []).append(bb)

        bk, bv = None, -1.0
        for k in [0, 1, 2, 3]:
            mdl, _ = run_sched(rtr, ytr, D, k, "C-koordinat", tro)
            v = float((pred_signed(mdl, rval) == yval).mean())
            if v > bv:
                bv, bk = v, k
        mdl, _ = run_sched(rtr, ytr, D, bk, "C-koordinat", tro)
        res.setdefault("BIZ: C-koordinat (4+k bit)", []).append(metrics(pred_signed(mdl, rte), yte)[0])
        cfg.setdefault("BIZ: C-koordinat (4+k bit)", []).append(bk)
        print("  seed %d bitti" % seed, flush=True)

    ours = np.array(res["BIZ: C-koordinat (4+k bit)"])
    print("", flush=True)
    for lbl in ["W1 sabit (genis izgara)", "W2 kuresel c/sqrt(t)",
                "W3 koordinat c/sqrt(n_i)", "BIZ: C-koordinat (4+k bit)"]:
        a = np.array(res[lbl])
        tag = ""
        if not lbl.startswith("BIZ"):
            d = ours - a
            sd = d.std(ddof=1)
            t = d.mean() / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
            mark = "  BERABERE" if abs(t) < 2.776 else ("  BIZ KAZANDIK" if t > 0 else "  BIZ KAYBETTIK")
            tag = " | biz-o: %+6.2f (t=%6.2f)%s" % (d.mean() * 100, t, mark)
        print("  %-28s %.4f +/-%.4f%s   sec: %s"
              % (lbl, a.mean(), a.std(), tag,
                 ["%g" % x for x in cfg[lbl]]), flush=True)


t0 = time.time()
study("MNIST 0-4 vs 5-9  (D-284'te +4.21 'zafer' vardi)", lambda s: DS.mnist()[1:])
print("  (%.1f dk)" % ((time.time() - t0) / 60), flush=True)
study("YOGUN SENTETIK  (D-284'te +4.07 'zafer' vardi)", make_synth)
print("  (%.1f dk)" % ((time.time() - t0) / 60), flush=True)
study("20NG comp vs rec (kontrol: burada zaten berabereydik)", lambda s: DS.LOADERS[0]()[1:])
print("\ntoplam sure %.1f dk" % ((time.time() - t0) / 60), flush=True)
