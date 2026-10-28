# -*- coding: utf-8 -*-
"""DENEY 281: TAMSAYI ARTIK-BIRIKTIRICI + ALGORITMANIN UC KUSURUNUN DUZELTMESI

D-280 teshisi: doygunluk YAYILMADIR. Belirleyici nicelik kutu/adim = etkin durum
sayisi. Master-float bunu 64 bitle cozuyor. Kac bit GERCEKTEN gerekli?

D-280'in ifsa ettigi UC KUSUR ve duzeltmeleri (hepsi TAMSAYI, sifir float):

  KUSUR 1 - Adim kuantuma kilitli: p<1 yapilamiyor (D-278'de optimal p=0.1 idi).
  FIX A   - ARTIK BIRIKTIRICI (error feedback / delta-sigma):
            k bitlik tamsayi artik r_i; hata aninda r += d*g; |r|>=2^k olunca
            TASIYIP sayaci 1 kaydir, r -= isaret*2^k.
            Efektif adim p = g/2^k, TAM, deterministik, tamsayi.
            k=0 -> aynen D-275 motoru (baseline).

  KUSUR 2 - Gurultulu etiket tam guclu guncelleme yapiyor: %5 etiket gurultusunde
            "guvenle yanlis" ornek buyuk olasilikla CEVRILMIS etikettir ama
            en buyuk hasari o verir.
  FIX B   - MARJ KAPISI: hata aninda |skor| > tau ise guncellemeyi ATLA.
            Tek tamsayi karsilastirmasi.

  KUSUR 3 - Sayaclar hic sonumlenmiyor; eski gurultu kalici birikiyor.
  FIX C   - SIZINTILI SONUMLEME: her K hatada e <- max(0, e-1) (global).
            0 tabani yuzunden bu UNIFORM bir olcekleme DEGILDIR (kucuk agirliklar
            olur, buyukler yarilanir) -> gercek seyreklestirme. Donanimda bedava.
"""
import math, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import run_mwu, pred_mwu, make_synth, load_real, LN2, ETA_GRID, metrics
from deney_275_tek_ray import pred_signed
from deney_278_float_master_vs_tamsayi import run_master

SEEDS = [0, 1, 2, 3, 4]
B_EXP = 3                      # 4 bit isaretli sayac (yogunda coken konfigurasyon)


def run_accum(rows, ys, D, B_exp, k, g, order, tau=None, decay_every=None):
    """4 bit sayac + k bit artik biriktirici (+ opsiyonel marj kapisi, sonumleme)."""
    EMAX = 2 ** B_exp - 1
    lut = np.array([0.0] + [2.0 ** i for i in range(EMAX)])
    thr = 2 ** k
    e = np.zeros(D, np.int16)
    s = np.ones(D, np.int8)
    r = np.zeros(D, np.int32)          # k bitlik artik
    mis = 0
    skipped = 0
    for t in order:
        a = rows[t]
        y = ys[t]
        sc = 0.0
        if len(a):
            sc = float((s[a] * lut[e[a]]).sum())
        pred = 1 if sc > 0 else 0
        if pred != y:
            mis += 1
            if tau is not None and abs(sc) > tau:      # FIX B: marj kapisi
                skipped += 1
                continue
            if len(a):
                d = 1 if y == 1 else -1
                r[a] += d * g                          # FIX A: biriktir
                ra = r[a]
                step = np.where(ra >= thr, 1, np.where(ra <= -thr, -1, 0)).astype(np.int8)
                r[a] = ra - step.astype(np.int32) * thr
                m = step != 0
                if m.any():
                    aa = a[m]
                    dd = step[m]
                    ea, sa = e[aa], s[aa]
                    zero = ea == 0
                    same = (~zero) & (sa == dd)
                    opp = (~zero) & (sa != dd)
                    ea = np.where(same, np.minimum(EMAX, ea + 1), ea)
                    ea = np.where(opp, ea - 1, ea)
                    ea = np.where(zero, 1, ea)
                    e[aa] = ea
                    s[aa] = np.where(zero, dd, sa).astype(np.int8)
            if decay_every and mis % decay_every == 0:  # FIX C: sizintili sonumleme
                e = np.maximum(0, e - 1)
    return ("signed", lut, e, s), mis, skipped


def tune(rows_tr, y_tr, D, rval, yval, tro, grid, runner):
    best, bv = None, -1.0
    for cfg in grid:
        m, _, _ = runner(cfg)
        v = float((pred_signed(m, rval) == yval).mean())
        if v > bv:
            bv, best = v, cfg
    return best


def study(name, get_data):
    print("\n" + "=" * 92, flush=True)
    print("### %s" % name, flush=True)
    print("=" * 92, flush=True)
    ref, mas, res, cfgs = [], [], {}, {}
    for seed in SEEDS:
        rtr, ytr, rte, yte, D = get_data(seed)
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(rtr))
        nv = int(0.2 * len(order))
        vi, tro = order[:nv], order[nv:]
        rval, yval = [rtr[i] for i in vi], ytr[vi]

        # referanslar
        be, bv = None, -1.0
        for eta in ETA_GRID:
            m, _ = run_mwu(rtr, ytr, D, eta, tro)
            v = float((pred_mwu(m, rval) == yval).mean())
            if v > bv:
                bv, be = v, eta
        m, _ = run_mwu(rtr, ytr, D, be, tro)
        ref.append(metrics(pred_mwu(m, rte), yte)[0])

        bp, bvm = None, -1.0
        for p in [1.0, 0.5, 0.25, 0.1, 0.05]:
            m, _ = run_master(rtr, ytr, D, B_EXP, p, tro)
            v = float((pred_signed(m, rval) == yval).mean())
            if v > bvm:
                bvm, bp = v, p
        m, _ = run_master(rtr, ytr, D, B_EXP, bp, tro)
        mas.append(metrics(pred_signed(m, rte), yte)[0])

        # FAZ 1: biriktirici bit taramasi
        for k in [0, 1, 2, 3, 4, 6]:
            gs = [2 ** j for j in range(k + 1)]
            bg = tune(rtr, ytr, D, rval, yval, tro, gs,
                      lambda g: run_accum(rtr, ytr, D, B_EXP, k, g, tro))
            m, _, _ = run_accum(rtr, ytr, D, B_EXP, k, bg, tro)
            res.setdefault(("A) biriktirici k=%d" % k, 4 + k), []).append(
                metrics(pred_signed(m, rte), yte)[0])
            cfgs.setdefault(k, []).append(bg)

        # FAZ 2: marj kapisi ve sonumleme (k=0, yani saf 4 bit uzerinde)
        taus = [None, 16384.0, 4096.0, 1024.0, 256.0]
        bt = tune(rtr, ytr, D, rval, yval, tro, taus,
                  lambda tau: run_accum(rtr, ytr, D, B_EXP, 0, 1, tro, tau=tau))
        m, _, sk = run_accum(rtr, ytr, D, B_EXP, 0, 1, tro, tau=bt)
        res.setdefault(("B) marj kapisi (4 bit)", 4), []).append(metrics(pred_signed(m, rte), yte)[0])

        decays = [None, 4000, 2000, 1000, 500]
        bd = tune(rtr, ytr, D, rval, yval, tro, decays,
                  lambda dk: run_accum(rtr, ytr, D, B_EXP, 0, 1, tro, decay_every=dk))
        m, _, _ = run_accum(rtr, ytr, D, B_EXP, 0, 1, tro, decay_every=bd)
        res.setdefault(("C) sonumleme (4 bit)", 4), []).append(metrics(pred_signed(m, rte), yte)[0])
        print("  seed %d bitti  (marj tau=%s, sonum K=%s)" % (seed, bt, bd), flush=True)

    R, M = np.array(ref), np.array(mas)
    print("\n  Warmuth FP64 (64 bit)     : %.2f%% +/-%.2f" % (R.mean() * 100, R.std() * 100))
    print("  Master-float (64+1 bit)   : %.2f%% +/-%.2f\n" % (M.mean() * 100, M.std() * 100))
    print("  %-26s %8s %10s %8s   %s" % ("model", "toplam bit", "acc%", "std", "master'a fark"))
    print("  " + "-" * 84)
    for key in sorted(res, key=lambda z: (z[0][0], z[1])):
        a = np.array(res[key])
        d = a - M
        sd = d.std(ddof=1)
        t = d.mean() / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
        mark = "  <-- BERABERE" if abs(t) < 2.776 else ""
        extra = ""
        if key[0].startswith("A) biriktirici"):
            kk = int(key[0].split("k=")[1])
            extra = "  g: %s" % cfgs[kk]
        print("  %-26s %8d %10.2f %8.2f   %+6.2f (t=%6.2f)%s%s"
              % (key[0], key[1], a.mean() * 100, a.std() * 100, d.mean() * 100, t, mark, extra))


study("YOGUN SENTETIK -- ASIL SINAV (4 bit burada %52'ye cokuyor)", make_synth)
study("GERCEK VERI 20 Newsgroups (seyrek)", load_real)
