# -*- coding: utf-8 -*-
"""KONTROL DENEYI: cevrimici (online) mistake-driven ogrenme KAOTIK midir?

Eger BIREBIR AYNI algoritmanin sadece hassasiyeti degistirilmis iki surumu
(fp32 vs fp64 vs longdouble) da birbirinden benzer oranda ayrisiyorsa,
log-uzayi surumunun ayrismasi ALGORITMIK bir fark degil, kaostur.
O zaman log-uzayi surumu Warmuth'un mesru (ve daha hassas) uygulamasidir.
"""
import math
import numpy as np

D, N, NT = 2000, 3000, 800
rng = np.random.RandomState(5)
ti = rng.choice(D, 20, replace=False)
ts = rng.choice([-1, 1], 20)
Xtr = rng.rand(N, D) < 0.02
ytr = ((Xtr[:, ti].astype(np.int8) @ ts) > 0).astype(np.int8)
Xte = rng.rand(NT, D) < 0.02
yte = ((Xte[:, ti].astype(np.int8) @ ts) > 0).astype(np.int8)
rows_tr = [np.where(r)[0] for r in Xtr]
Xtr8 = Xtr.astype(np.int8)
Xte8 = Xte.astype(np.int8)


def original(eta, dt):
    wp = np.ones(D, dt) / (2 * D)
    wn = wp.copy()
    flips = []
    for t in range(N):
        pred = 1 if np.dot(wp - wn, Xtr8[t]) > 0 else 0
        flips.append(pred)
        if pred != ytr[t]:
            a = rows_tr[t]
            if len(a):
                if pred > ytr[t]:
                    wp[a] *= math.exp(-eta); wn[a] *= math.exp(eta)
                else:
                    wp[a] *= math.exp(eta); wn[a] *= math.exp(-eta)
                tot = wp.sum() + wn.sum()
                wp /= tot; wn /= tot
    return (Xte8 @ (wp - wn) > 0).astype(np.int8), np.array(flips, np.int8)


print("AYNI ALGORITMA, SADECE HASSASIYET FARKLI (fp32 vs fp64 vs float128):")
print("%-8s | %-22s | %-22s | %s"
      % ("eta", "fp32 vs fp64 uyum", "fp64 vs f128 uyum", "ilk ayrisma adimi (fp32/fp64)"))
for eta in [0.1, 0.5, math.log(2), 1.0, 2.0]:
    a32, f32 = original(eta, np.float32)
    a64, f64 = original(eta, np.float64)
    aLD, fLD = original(eta, np.longdouble)
    d = np.where(f32 != f64)[0]
    first = int(d[0]) if len(d) else -1
    print("%-8.4f | %8d fark %6.2f%%  | %8d fark %6.2f%%  | %d / %d adim icinde"
          % (eta,
             int((a32 != a64).sum()), (a32 == a64).mean() * 100,
             int((a64 != aLD).sum()), (a64 == aLD).mean() * 100,
             first, N))
print("\nSONUC: ayni algoritmanin iki hassasiyeti bile ayrisiyorsa, sistem kaotiktir;")
print("log-uzayi surumunun ayrismasi algoritmik bir sapma degildir.")
