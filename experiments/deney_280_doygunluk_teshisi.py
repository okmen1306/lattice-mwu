# -*- coding: utf-8 -*-
"""DENEY 280 (TESHIS): DOYGUNLUK -- ORTAK KAYMA MI, YAYILMA MI?

Iddia: "eksik 4. parca normalizasyondur; sayaclar tavana yapisiyor, bagil
normalizasyon bunu cozer."

Normalizasyon (ortak ofset cikarmak) YALNIZCA sayaclarin hepsi ayni yone
suruklendiginde (ORTAK KAYMA / common-mode drift) ise yarar. Eger sayaclar
iki uca birden dagiliyorsa (YAYILMA / spread), ortak ofset hicbir sey cozmez.

Bu deney son sayac dagilimini olcer ve iki senaryoyu ayirt eder:
  ortak kayma  -> ortalama |ortalama| buyuk, uclara dagilim tek tarafli
  yayilma      -> ortalama ~ 0, ama HER IKI uca da yigilma var
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d274 import make_synth, load_real, LN2
from d275_signed import run_signed


def diagnose(name, get_data, B_exp):
    EMAX = 2 ** B_exp - 1
    print("\n" + "=" * 78)
    print("### %s   (tek ray, %d bit, EMAX=%d)" % (name, B_exp + 1, EMAX))
    print("=" * 78)
    for seed in [0, 1]:
        rtr, ytr, rte, yte, D = get_data(seed)
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(rtr))
        tro = order[int(0.2 * len(order)):]
        (_, lut, e, s), mis = run_signed(rtr, ytr, D, B_exp, LN2, tro)

        # isaretli net durum: pozitif rayda +e, negatif rayda -e
        net = s.astype(np.int32) * e.astype(np.int32)
        touched = e > 0                      # hic guncellenmis sinapslar
        at_max = int((e == EMAX).sum())
        at_zero = int((e == 0).sum())
        pos_max = int(((e == EMAX) & (s > 0)).sum())
        neg_max = int(((e == EMAX) & (s < 0)).sum())

        print("\n  seed %d | hata=%d | dokunulan sinaps=%d/%d" % (seed, mis, touched.sum(), D))
        print("    tavana yapisan (e=EMAX) : %d  (+ tarafta %d, - tarafta %d)"
              % (at_max, pos_max, neg_max))
        print("    olu (e=0)               : %d" % at_zero)
        print("    net durum: ortalama=%.3f  medyan=%.1f  std=%.3f"
              % (net.mean(), np.median(net), net.std()))
        print("    |ortalama| / std        : %.4f   <-- 1'e yakin=ORTAK KAYMA, 0'a yakin=YAYILMA"
              % (abs(net.mean()) / max(net.std(), 1e-12)))
        hist = np.bincount(net + EMAX, minlength=2 * EMAX + 1)
        print("    net histogram (-%d..+%d): %s" % (EMAX, EMAX, list(hist)))
        if at_max > 0:
            bal = min(pos_max, neg_max) / max(max(pos_max, neg_max), 1)
            print("    tavan dengesi min/max   : %.3f   <-- 1'e yakin=IKI UCA BIRDEN (yayilma)" % bal)


diagnose("YOGUN SENTETIK (entropi burada cokuyor)", make_synth, 3)   # 4 bit
diagnose("GERCEK VERI 20NG (burada calisiyor)", load_real, 3)        # 4 bit
