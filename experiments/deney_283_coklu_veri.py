# -*- coding: utf-8 -*-
"""DENEY 283: COKLU GERCEK VERI SETI + U-YASASININ KOR SINAVI

EKSIK #1: tum gercek-veri iddialarimiz TEK veri setine (20NG comp vs rec)
dayaniyor -> veri seti duzeyinde n=1.

Bu deney:
 (a) bes farkli gercek gorevde 4-bit tamsayi motoru vs ayarlanmis FP64 Warmuth
 (b) D-282'nin yasasinin KOR SINAVI: her veri setinde once U=M*A/D olculur,
     B* = 1/2 log2(U) + 2.6 TAHMIN edilir, SONRA gercek B* olculur.
     Yasa bu noktalarin hicbirini gormedi.
"""
import math, sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deney_274_temiz_tur import run_mwu, pred_mwu, to_rows, metrics
from deney_275_tek_ray import run_signed, pred_signed

LN2 = math.log(2.0)
SEEDS = [0, 1, 2]
B_GRID = [2, 3, 4, 5, 6, 8]
ETA_GRID = [0.1, 0.5, LN2, 1.0, 2.0]
MAXF = 10000


def _vec(tr_txt, te_txt):
    from sklearn.feature_extraction.text import CountVectorizer
    v = CountVectorizer(binary=True, max_features=MAXF, min_df=2, stop_words="english")
    return v.fit_transform(tr_txt), v.transform(te_txt)


def ng(pos, neg, name):
    from sklearn.datasets import fetch_20newsgroups
    cats = pos + neg
    tr = fetch_20newsgroups(subset="train", categories=cats,
                            remove=("headers", "footers", "quotes"), shuffle=False)
    te = fetch_20newsgroups(subset="test", categories=cats,
                            remove=("headers", "footers", "quotes"), shuffle=False)
    lab = lambda t: np.array([1 if t.target_names[i] in pos else 0 for i in t.target], np.int8)
    Xtr, Xte = _vec(tr.data, te.data)
    return name, to_rows(Xtr), lab(tr), to_rows(Xte), lab(te), Xtr.shape[1]


def rcv1():
    from sklearn.datasets import fetch_rcv1
    d = fetch_rcv1(subset="train")                 # ~23k dok, 47k ozellik
    X = (d.data[:, :MAXF] if d.data.shape[1] > MAXF else d.data)
    X = (X > 0).astype(np.int8).tocsr()
    ccat = list(d.target_names).index("CCAT")
    y = np.asarray(d.target[:, ccat].todense()).ravel().astype(np.int8)
    n = X.shape[0]
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    ntr = int(0.7 * n)
    return ("RCV1 CCAT vs rest", to_rows(X[idx[:ntr]]), y[idx[:ntr]],
            to_rows(X[idx[ntr:]]), y[idx[ntr:]], X.shape[1])


def mnist():
    from sklearn.datasets import fetch_openml
    d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = (d.data > 127).astype(np.int8)             # ikilestir
    y = (d.target.astype(int) >= 5).astype(np.int8)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y))[:30000]
    X, y = X[idx], y[idx]
    from scipy import sparse
    Xs = sparse.csr_matrix(X)
    ntr = 21000
    return ("MNIST 0-4 vs 5-9 (ikili piksel)", to_rows(Xs[:ntr]), y[:ntr],
            to_rows(Xs[ntr:]), y[ntr:], X.shape[1])


LOADERS = [
    lambda: ng(["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
                "comp.sys.mac.hardware", "comp.windows.x"],
               ["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"],
               "20NG comp vs rec (mevcut)"),
    lambda: ng(["sci.crypt", "sci.electronics", "sci.med", "sci.space"],
               ["talk.politics.guns", "talk.politics.mideast", "talk.politics.misc",
                "talk.religion.misc"], "20NG sci vs talk"),
    lambda: ng(["alt.atheism", "soc.religion.christian"],
               ["misc.forsale", "rec.autos"], "20NG din vs ticaret/oto"),
    rcv1,
    mnist,
]


def evaluate(name, rtr, ytr, rte, yte, D):
    accs = {B: [] for B in B_GRID}
    wref, majs, Us, As, Ms = [], [], [], [], []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(rtr))
        nv = int(0.2 * len(order))
        vi, tro = order[:nv], order[nv:]
        rval, yval = [rtr[i] for i in vi], ytr[vi]
        majs.append(max(yte.mean(), 1 - yte.mean()))
        A = float(np.mean([len(r) for r in rtr])); As.append(A)

        be, bv = None, -1.0
        for eta in ETA_GRID:
            m, _ = run_mwu(rtr, ytr, D, eta, tro)
            v = float((pred_mwu(m, rval) == yval).mean())
            if v > bv:
                bv, be = v, eta
        m, _ = run_mwu(rtr, ytr, D, be, tro)
        wref.append(metrics(pred_mwu(m, rte), yte)[0])

        for B in B_GRID:
            mdl, mis = run_signed(rtr, ytr, D, B - 1, LN2, tro)
            accs[B].append(metrics(pred_signed(mdl, rte), yte)[0])
            if B == 4:
                Ms.append(mis); Us.append(mis * A / D)

    maj, w = float(np.mean(majs)), float(np.mean(wref))
    span = max(w - maj, 1e-9)
    curve = {B: float(np.mean(accs[B])) for B in B_GRID}
    sd4 = float(np.std(accs[4]))
    Bstar = next((B for B in B_GRID if (curve[B] - maj) / span >= 0.90), None)
    U = float(np.mean(Us))
    pred = 0.5 * math.log2(max(U, 1.0001)) + 2.6
    print("\n" + "-" * 96, flush=True)
    print("### %s   (D=%d, A=%.1f, egitim=%d, test=%d)" % (name, D, np.mean(As), len(rtr), len(rte)), flush=True)
    print("  cogunluk tabani : %.4f" % maj, flush=True)
    print("  Warmuth FP64    : %.4f" % w, flush=True)
    print("  4-bit tamsayi   : %.4f +/-%.4f   (Warmuth'a %+.2f puan)"
          % (curve[4], sd4, (curve[4] - w) * 100), flush=True)
    print("  normalize skor z: " + "  ".join("B=%d:%.3f" % (B, (curve[B] - maj) / span) for B in B_GRID), flush=True)
    print("  >> U = M*A/D = %.2f  (M=%.0f)" % (U, np.mean(Ms)), flush=True)
    print("  >> YASA TAHMINI B* = 0.5*log2(U)+2.6 = %.2f  ->  %d" % (pred, max(2, round(pred))), flush=True)
    print("  >> OLCULEN     B* = %s" % (Bstar if Bstar else ">8"), flush=True)
    ok = (Bstar is not None) and abs(Bstar - pred) <= 1.0
    print("  >> %s" % ("TUTTU (|fark|<=1)" if ok else "TUTMADI"), flush=True)
    return dict(name=name, maj=maj, w=w, a4=curve[4], sd4=sd4, U=U, pred=pred, Bstar=Bstar, ok=ok)


if __name__ == "__main__":
    t0 = time.time()
    rows = []
    print("=" * 96, flush=True)
    print("### DENEY 283 -- COKLU GERCEK VERI + U-YASASININ KOR SINAVI", flush=True)
    print("=" * 96, flush=True)
    for L in LOADERS:
        try:
            name, rtr, ytr, rte, yte, D = L()
        except Exception as ex:
            print("\n  [ATLANDI] %s: %s" % (getattr(L, "__name__", "veri"), type(ex).__name__), flush=True)
            continue
        try:
            rows.append(evaluate(name, rtr, ytr, rte, yte, D))
        except Exception as ex:
            print("  [HATA] %s: %s %s" % (name, type(ex).__name__, ex), flush=True)
        print("  (%.1f dk)" % ((time.time() - t0) / 60), flush=True)

    print("\n" + "=" * 96, flush=True)
    print("### OZET", flush=True)
    print("=" * 96, flush=True)
    print("%-34s %8s %8s %9s %8s %7s %7s %6s" %
          ("veri seti", "cogunl", "warmuth", "4-bit", "fark", "U", "tahmin", "olcum"), flush=True)
    for r in rows:
        print("%-34s %8.4f %8.4f %9.4f %+8.2f %7.1f %7.2f %6s%s" %
              (r["name"][:34], r["maj"], r["w"], r["a4"], (r["a4"] - r["w"]) * 100,
               r["U"], r["pred"], r["Bstar"] if r["Bstar"] else ">8",
               "  OK" if r["ok"] else "  X"), flush=True)
    hit = sum(1 for r in rows if r["ok"])
    print("\nU-YASASI KOR SINAVI: %d/%d tuttu" % (hit, len(rows)), flush=True)
    print("toplam sure %.1f dk" % ((time.time() - t0) / 60), flush=True)
