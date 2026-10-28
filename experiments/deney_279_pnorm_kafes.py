# -*- coding: utf-8 -*-
"""DENEY 279: p-NORM CANAGININ KAFES HALI

Ana formul:  grad_phi(w_{t+1}) = grad_phi(w_t) - eta * grad_L
Dual kafese oturunca primal ayna ters donusumu bir LUT'a iner. Bu, HER canak icin
gecerlidir; is, hangi canakta kazanc oldugudur.

  Oklid   : dual = w        -> LUT = kimlik      -> oldurulecek exp yok
  Entropi : dual = ln w     -> LUT = 2^k         -> BIZIM MOTOR (D-274..278)
  p-Norm  : dual = sgn|w|^{p-1} -> LUT = sgn|s|^r -> BU DENEY
  Matris  : dual = ln W     -> ozdegerlerde kaydirma -> O(d^3) baskin

NEDEN p-NORM:
D-278, yogun veride tamsayi-saf ogrenmenin coktugunu gosterdi (52.43 vs 75.27),
cunku ENTROPI bagi ustel duyarlidir: bir kafes adimi agirligi 2 KATINA cikarir,
ve optimal adim p=0.1 idi (5/5 tohum) -- yani kuantum alti bilgi kayboluyordu.
p-Norm bagi POLINOMIYALDIR: s->s+1 orani ((s+1)/s)^r, s buyudukce 1'e yaklasir.
Dahasi p-norm canaginda dual zaten bir GRADYAN BIRIKTIRICISIDIR; dogal kafes tam
sayimdir, yani adim tanim geregi 1 kafes birimidir -> kuantum alti problem DOGMAZ.

NORMALIZATOR DUSER: w_i = sgn(t_i)|t_i|^{q-1} / ||t||_q^{q-2}; payda tum i icin
ayni POZITIF skalerdir, isaret tabanli tahminde tamamen duser -> O(D) global
normalizasyon YOK.

MOTOR:
  durum      : isaretli tamsayi sayac s_i in [-S, S], S = 2^{B-1} - 1  (B bit)
  guncelleme : hata aninda s[aktif] += d,  d = +1 (y=1) / -1 (y=0), kirpmali
  tahmin     : sign( sum_{aktif} sgn(s_i) * |s_i|^r )
  r = 1 -> PERCEPTRON (Oklid canagi). r buyudukce entropi tarafina kopru.
"""
import math, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import run_mwu, pred_mwu, make_synth, load_real, LN2, ETA_GRID, metrics
from deney_275_tek_ray import run_signed, pred_signed
from deney_278_float_master_vs_tamsayi import run_master

SEEDS = [0, 1, 2, 3, 4]
R_GRID = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
BITS = [4, 6, 8]


# ---------------------------------------------- p-NORM KAFES MOTORU
def run_pnorm(rows, ys, D, B, r, order):
    S = 2 ** (B - 1) - 1                       # B bit isaretli sayac
    k = np.arange(-S, S + 1, dtype=np.int64)
    lut = np.sign(k) * np.abs(k).astype(np.float64) ** r   # LUT[s+S]
    s = np.zeros(D, np.int64)
    mis = 0
    for t in order:
        a = rows[t]
        y = ys[t]
        pred = 0
        if len(a):
            pred = 1 if float(lut[s[a] + S].sum()) > 0 else 0
        if pred != y:
            mis += 1
            if len(a):
                s[a] = np.clip(s[a] + (1 if y == 1 else -1), -S, S)
    return ("pnorm", lut, s, S), mis


def pred_pnorm(model, rows):
    _, lut, s, S = model
    out = np.zeros(len(rows), np.int8)
    for i, a in enumerate(rows):
        if len(a):
            out[i] = 1 if float(lut[s[a] + S].sum()) > 0 else 0
    return out


def study(name, get_data):
    print("\n" + "=" * 92, flush=True)
    print("### %s" % name, flush=True)
    print("=" * 92, flush=True)
    ref, res, chosen = [], {}, {}
    for seed in SEEDS:
        rtr, ytr, rte, yte, D = get_data(seed)
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(rtr))
        nv = int(0.2 * len(order))
        vi, tro = order[:nv], order[nv:]
        rval, yval = [rtr[i] for i in vi], ytr[vi]

        # A) Warmuth FP64 referans
        be, bv = None, -1.0
        for eta in ETA_GRID:
            m, _ = run_mwu(rtr, ytr, D, eta, tro)
            v = float((pred_mwu(m, rval) == yval).mean())
            if v > bv:
                bv, be = v, eta
        m, _ = run_mwu(rtr, ytr, D, be, tro)
        ref.append(metrics(pred_mwu(m, rte), yte)[0])

        for B in BITS:
            # p-NORM kafes: r validation'da ayarlanir
            br, bvr = None, -1.0
            for r in R_GRID:
                m, _ = run_pnorm(rtr, ytr, D, B, r, tro)
                v = float((pred_pnorm(m, rval) == yval).mean())
                if v > bvr:
                    bvr, br = v, r
            m, _ = run_pnorm(rtr, ytr, D, B, br, tro)
            res.setdefault(("p-NORM kafes", B), []).append(metrics(pred_pnorm(m, rte), yte)[0])
            chosen.setdefault(B, []).append(br)

            # ENTROPI kafes (bizim eski motor), ayni bit butcesi
            m, _ = run_signed(rtr, ytr, D, B - 1, LN2, tro)
            res.setdefault(("ENTROPI kafes (eski)", B), []).append(
                metrics(pred_signed(m, rte), yte)[0])

            # MASTER-FLOAT (BitNet/QAT analogu), ayni cikarim
            bp, bvm = None, -1.0
            for p in [1.0, 0.5, 0.25, 0.1, 0.05]:
                m, _ = run_master(rtr, ytr, D, B - 1, p, tro)
                v = float((pred_signed(m, rval) == yval).mean())
                if v > bvm:
                    bvm, bp = v, p
            m, _ = run_master(rtr, ytr, D, B - 1, bp, tro)
            res.setdefault(("MASTER-FLOAT (64 bit)", B), []).append(
                metrics(pred_signed(m, rte), yte)[0])
        print("  seed %d bitti" % seed, flush=True)

    R = np.array(ref)
    print("\n  A) Warmuth FP64 (eta* ayarli, 64 bit/sinaps): %.2f%% +/-%.2f\n"
          % (R.mean() * 100, R.std() * 100), flush=True)
    for B in BITS:
        print("  --- %d BIT ---" % B, flush=True)
        base = np.array(res[("MASTER-FLOAT (64 bit)", B)])
        for arm in ["p-NORM kafes", "ENTROPI kafes (eski)", "MASTER-FLOAT (64 bit)"]:
            a = np.array(res[(arm, B)])
            tag = ""
            if arm != "MASTER-FLOAT (64 bit)":
                d = a - R
                sd = d.std(ddof=1)
                t = d.mean() / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
                tag = "  Warmuth'a %+6.2f (t=%6.2f)%s" % (
                    d.mean() * 100, t, "  <-- BERABERE" if abs(t) < 2.776 else "")
            extra = ""
            if arm == "p-NORM kafes":
                extra = "   secilen r: %s" % ["%g" % x for x in chosen[B]]
            print("    %-24s %8.2f +/-%-5.2f%s%s"
                  % (arm, a.mean() * 100, a.std() * 100, tag, extra), flush=True)
        print("", flush=True)


study("SENTETIK DENGELI (yogun) -- ASIL SINAV: entropi burada cokmustu", make_synth)
study("GERCEK VERI -- 20 Newsgroups (seyrek)", load_real)
