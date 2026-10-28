# -*- coding: utf-8 -*-
"""DENEY 274 -- D273'un temiz turu.
Duzeltilen 4 kusur: (1) dengeli veri, (2) GERCEK veri seti, (3) 5 seed,
(4) baseline validation'da ayarlandi. Test setine sadece bir kez bakilir.
"""
import math, time
import numpy as np
from scipy import sparse

LN2 = math.log(2.0)
SEEDS = [0, 1, 2, 3, 4]
ETA_GRID = [0.05, 0.1, 0.25, 0.5, LN2, 1.0, 2.0, 4.0]
B_GRID = [2, 3, 4, 5, 6]


# ---------------------------------------------------------------- ogrenciler
def run_mwu(rows, ys, D, eta, order):
    """Warmuth MWU, log-uzayinda BIREBIR ayni algoritma.
    Normalizasyon global bir olcek carpanidir, skorun ISARETINI degistirmez;
    log-uzayina tasimak overflow/underflow riskini tamamen kaldirir."""
    lp = np.zeros(D)
    lnn = np.zeros(D)
    mis = 0
    for t in order:
        a = rows[t]
        y = ys[t]
        if len(a):
            u = lp[a]
            v = lnn[a]
            m = max(u.max(), v.max())
            s = np.exp(u - m).sum() - np.exp(v - m).sum()
            pred = 1 if s > 0 else 0
        else:
            pred = 0
        if pred != y:
            mis += 1
            if len(a):
                if y == 1:
                    lp[a] += eta
                    lnn[a] -= eta
                else:
                    lp[a] -= eta
                    lnn[a] += eta
    return ("mwu", lp, lnn), mis


def pred_mwu(model, rows):
    _, lp, lnn = model
    out = np.zeros(len(rows), np.int8)
    for i, a in enumerate(rows):
        if len(a):
            u = lp[a]
            v = lnn[a]
            m = max(u.max(), v.max())
            out[i] = 1 if (np.exp(u - m).sum() - np.exp(v - m).sum()) > 0 else 0
    return out


def run_quant(rows, ys, D, B, eta, order):
    """Muhammed'in motoru, genellestirilmis.
    B=4 ve eta=ln2 -> LUT = [0,1,2,4,...,16384] = D273'un birebir bit-shift motoru.
    Durum 0 = olu sinaps (SAT kisiti)."""
    S = 2 ** B - 1
    lut = np.array([0.0] + [math.exp(i * eta) for i in range(S)])
    cp = np.ones(D, np.int16)
    cn = np.ones(D, np.int16)
    mis = 0
    for t in order:
        a = rows[t]
        y = ys[t]
        if len(a):
            pred = 1 if (lut[cp[a]].sum() - lut[cn[a]].sum()) > 0 else 0
        else:
            pred = 0
        if pred != y:
            mis += 1
            if len(a):
                if y == 1:
                    cp[a] = np.minimum(S, cp[a] + 1)
                    cn[a] = np.maximum(0, cn[a] - 1)
                else:
                    cp[a] = np.maximum(0, cp[a] - 1)
                    cn[a] = np.minimum(S, cn[a] + 1)
    return ("quant", lut, cp, cn), mis


def pred_quant(model, rows):
    _, lut, cp, cn = model
    out = np.zeros(len(rows), np.int8)
    for i, a in enumerate(rows):
        if len(a):
            out[i] = 1 if (lut[cp[a]].sum() - lut[cn[a]].sum()) > 0 else 0
    return out


def run_perc(rows, ys, D, order):
    """Klasik toplamsal Perceptron -- carpimsal guncellemenin bir sey katip
    katmadigini olcen capa."""
    w = np.zeros(D)
    mis = 0
    for t in order:
        a = rows[t]
        y = ys[t]
        pred = 1 if (len(a) and w[a].sum() > 0) else 0
        if pred != y:
            mis += 1
            if len(a):
                w[a] += 1 if y == 1 else -1
    return ("perc", w), mis


def pred_perc(model, rows):
    _, w = model
    return np.array([1 if (len(a) and w[a].sum() > 0) else 0 for a in rows], np.int8)


# ---------------------------------------------------------------- metrikler
def metrics(p, y):
    acc = float((p == y).mean())
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return acc, f1, 0.5 * (rc + tnr)


def mcnemar(a, b, y):
    ca = a == y
    cb = b == y
    n01 = int((ca & ~cb).sum())
    n10 = int((~ca & cb).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    from scipy.stats import binomtest
    return n01, n10, float(binomtest(min(n01, n10), n, 0.5).pvalue)


def to_rows(X):
    X = sparse.csr_matrix(X)
    return [X.indices[X.indptr[i]:X.indptr[i + 1]].astype(np.int64)
            for i in range(X.shape[0])]


# ---------------------------------------------------------------- veri
_REAL_CACHE = {}


def load_real(seed=None):
    if "d" in _REAL_CACHE:
        return _REAL_CACHE["d"]
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    COMP = ["comp.graphics", "comp.os.ms-windows.misc",
            "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
            "comp.windows.x"]
    REC = ["rec.autos", "rec.motorcycles", "rec.sport.baseball",
           "rec.sport.hockey"]
    cats = COMP + REC
    # headers/footers/quotes CIKARILDI -> 20NG'nin meshur sizinti kaynagi kapali
    tr = fetch_20newsgroups(subset="train", categories=cats,
                            remove=("headers", "footers", "quotes"),
                            shuffle=False)
    te = fetch_20newsgroups(subset="test", categories=cats,
                            remove=("headers", "footers", "quotes"),
                            shuffle=False)

    def lab(t):
        return np.array([1 if t.target_names[i] in COMP else 0
                         for i in t.target], np.int8)

    vec = CountVectorizer(binary=True, max_features=10000, min_df=2,
                          stop_words="english")
    Xtr = vec.fit_transform(tr.data)   # fit SADECE train uzerinde
    Xte = vec.transform(te.data)
    _REAL_CACHE["d"] = (to_rows(Xtr), lab(tr), to_rows(Xte), lab(te),
                        Xtr.shape[1])
    return _REAL_CACHE["d"]


def make_synth(seed, D=10000, K=20, p=0.10, n_tr=12000, n_te=3000,
               noise=0.05):
    rng = np.random.RandomState(1000 + seed)
    ti = rng.choice(D, K, replace=False)
    ts = np.array([1] * (K // 2) + [-1] * (K // 2))   # DENGELI isaretler
    rng.shuffle(ts)

    def gen(n):
        rows = []
        ys = np.zeros(n, np.int8)
        done = 0
        while done < n:
            c = min(1000, n - done)                     # bellek icin chunk
            M = rng.rand(c, D) < p
            s = M[:, ti].astype(np.int8) @ ts
            y = (s > 0).astype(np.int8)
            tie = s == 0
            y[tie] = (rng.rand(int(tie.sum())) < 0.5).astype(np.int8)  # yazi-tura
            fl = rng.rand(c) < noise
            y[fl] = 1 - y[fl]
            ys[done:done + c] = y
            rows.extend(to_rows(sparse.csr_matrix(M)))
            done += c
        return rows, ys

    rtr, ytr = gen(n_tr)
    rte, yte = gen(n_te)
    return rtr, ytr, rte, yte, D


# ---------------------------------------------------------------- deney
def experiment(name, get_data):
    print("\n" + "#" * 78, flush=True)
    print("### VERI SETI: %s" % name, flush=True)
    print("#" * 78, flush=True)
    agg, pooled, pooled_y = {}, {}, []

    for seed in SEEDS:
        rtr, ytr, rte, yte, D = get_data(seed)
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(rtr))
        n_val = int(0.2 * len(order))
        val_idx, tr_order = order[:n_val], order[n_val:]   # val disarida tutuldu
        rval = [rtr[i] for i in val_idx]
        yval = ytr[val_idx]

        # --- hiperparametre ayari: SADECE validation, test'e bakilmaz ---
        best_eta, bv = None, -1.0
        for eta in ETA_GRID:
            m, _ = run_mwu(rtr, ytr, D, eta, tr_order)
            v = float((pred_mwu(m, rval) == yval).mean())
            if v > bv:
                bv, best_eta = v, eta
        best_bq, bvq = None, -1.0
        for B in B_GRID:
            for eta in ETA_GRID:
                m, _ = run_quant(rtr, ytr, D, B, eta, tr_order)
                v = float((pred_quant(m, rval) == yval).mean())
                if v > bvq:
                    bvq, best_bq = v, (B, eta)

        res = {}
        maj = 1 if yval.mean() > 0.5 else 0
        res["M1 Cogunluk (sifir zeka)"] = (np.full(len(rte), maj, np.int8), 0, "-")
        m, mis = run_perc(rtr, ytr, D, tr_order)
        res["M2 Perceptron (toplamsal)"] = (pred_perc(m, rte), mis, "-")
        m, mis = run_mwu(rtr, ytr, D, best_eta, tr_order)
        res["M3 Warmuth MWU (eta*)"] = (pred_mwu(m, rte), mis, "eta=%.3g" % best_eta)
        m, mis = run_quant(rtr, ytr, D, 4, LN2, tr_order)
        res["M4 BitShift 4bit (D273)"] = (pred_quant(m, rte), mis, "B=4,eta=ln2")
        B, eta = best_bq
        m, mis = run_quant(rtr, ytr, D, B, eta, tr_order)
        res["M5 QuantMWU (B*,eta*)"] = (pred_quant(m, rte), mis,
                                        "B=%d,eta=%.3g" % (B, eta))

        print("\n-- seed %d | test n=%d | cogunluk tabani %.2f%% --"
              % (seed, len(rte), max(yte.mean(), 1 - yte.mean()) * 100), flush=True)
        print("   %-28s %7s %7s %7s %11s  %s"
              % ("model", "acc%", "F1", "bal%", "online-err", "hiper"), flush=True)
        for k in sorted(res):
            p, mis, hp = res[k]
            acc, f1, bal = metrics(p, yte)
            print("   %-28s %7.2f %7.3f %7.2f %11s  %s"
                  % (k, acc * 100, f1, bal * 100, mis if mis else "-", hp), flush=True)
            agg.setdefault(k, []).append((acc, f1, bal, mis))
            pooled.setdefault(k, []).append(p)
        pooled_y.append(yte)

    print("\n" + "=" * 78, flush=True)
    print("OZET (%d seed, ortalama +/- std) -- %s" % (len(SEEDS), name), flush=True)
    print("=" * 78, flush=True)
    print("%-28s %17s %15s %12s" % ("model", "test acc%", "F1", "online-err"),
          flush=True)
    for k in sorted(agg):
        a = np.array(agg[k])
        print("%-28s %8.2f +/-%-6.2f %6.3f +/-%-6.3f %10.1f"
              % (k, a[:, 0].mean() * 100, a[:, 0].std() * 100,
                 a[:, 1].mean(), a[:, 1].std(), a[:, 3].mean()), flush=True)

    yy = np.concatenate(pooled_y)
    pm3 = np.concatenate(pooled["M3 Warmuth MWU (eta*)"])
    a3 = np.array(agg["M3 Warmuth MWU (eta*)"])[:, 0]
    for ch in ["M4 BitShift 4bit (D273)", "M5 QuantMWU (B*,eta*)"]:
        pc = np.concatenate(pooled[ch])
        n01, n10, pv = mcnemar(pm3, pc, yy)
        d = np.array(agg[ch])[:, 0] - a3
        sd = d.std(ddof=1)
        tstat = d.mean() / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
        print("\n[%s]  vs  [M3 Warmuth MWU(eta*)]" % ch, flush=True)
        print("   seed-basi fark: %+.2f puan (std %.2f) | eslesmis t=%.2f (df=%d)"
              % (d.mean() * 100, sd * 100, tstat, len(d) - 1), flush=True)
        print("   McNemar (havuzlanmis n=%d): n01=%d n10=%d p=%.4f"
              % (len(yy), n01, n10, pv), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    experiment("GERCEK VERI -- 20 Newsgroups comp.* vs rec.* (ikili BoW, D=10000)",
               load_real)
    experiment("SENTETIK DENGELI -- D=10000, K=20 (10+/10-), p=0.1, %5 gurultu",
               make_synth)
    print("\ntoplam sure %.1f dk" % ((time.time() - t0) / 60), flush=True)
