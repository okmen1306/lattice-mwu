# Lattice-Quantized Multiplicative Weights

**How many bits does an online linear learner need?**

Multiplicative weights (MWU / Hedge) for online linear classification, with the dual variable θ = ln w restricted to an integer lattice δℤ inside a bounded box. Every weight becomes a power of two and every update becomes an integer increment, i.e. a bit shift. The learner keeps no floating-point master copy and no optimizer moments, and needs no backward pass. Its whole state is 4 + k integer bits per weight.

**Paper:** [paper/kafes_mwu.pdf](paper/kafes_mwu.pdf) (Turkish, with an English abstract), Zenodo DOI [10.5281/zenodo.23568755](https://doi.org/10.5281/zenodo.23568755)
**Author:** Muhammed Ökmen ([ORCID 0009-0002-9709-966X](https://orcid.org/0009-0002-9709-966X))

## Main results

- **Theory.**
  - A regret bound E[L] ≤ c(δ)·L* + c(δ)·ln D / η, with c(δ) = δ / (1 − e^(−δ)). Bernoulli stepping decouples the lattice spacing δ from the effective learning rate η.
  - At fixed δ = ln 2 the bound is only constant-competitive: a pure `1<<k` learner and no-regret cannot be had at the same time.
  - **Exactness lemma** for an integer residual accumulator: with θ̃ = θ + ρ, θ̃ follows exact mirror descent while |ρ| < δ. The bounded box is the thing error feedback repairs, not something the theory fails to cover.
- **A 4-bit learner matches a tuned FP64 MWU** on four sparse real tasks (three 20 Newsgroups splits and RCV1). On 20 Newsgroups: 90.44 ± 0.46 vs 90.31 ± 0.46.
- **7 bits ≡ 65 bits.** A 4-bit counter plus a 3-bit integer residual accumulator matches a 65-bit float-master learner on dense data (−1.11, t = −2.28), a 9.3× state reduction. At 8 bits it is more stable than the master (±0.20 vs ±0.61).
- **Bit-width law.** B* ≳ ½·log₂ U, where U is the per-coordinate update count. The total horizon T is the wrong variable; the T-form is refuted experimentally.

## Negative results

The paper reports these openly:

- **No clear win over MWU.** Against Warmuth's published algorithm the integer learner is at worst tied and ahead on two benchmarks. Against MWU + AdaGrad it ties on two of three and is behind on MNIST.
- **Two apparent victories were destroyed by our own controls.** One came from a broken majority baseline, the other from a 1.78 × 10⁻¹⁵ lookup-table difference.
- **Online learning here is chaotic.** The measured chaos band is 0.90 points on 20 Newsgroups but 4.53 points on MNIST, so the MNIST gap is below the minimum detectable effect.
- **The choice of potential (Euclidean, p-norm or entropy) made no measurable difference.**
- **No contribution is verified as novel.** Winnow2, fixed-β Weighted Majority, LNS training, 8-bit optimizer state, error feedback and AdaGrad all came earlier. The paper is a controlled measurement with its boundaries mapped.

The integer residual accumulator from this work later became the update rule of [ZeroFloat](https://github.com/okmen1306/zerofloat), fully integer training of language models up to 1B parameters.

## Running the experiments

```bash
pip install -r requirements.txt
cd experiments
python deney_274_temiz_tur.py        # main comparison (D-274)
```

Datasets are downloaded by scikit-learn (20 Newsgroups, RCV1, MNIST). Every experiment has its raw terminal output next to it (`*_ham_cikti.txt`). Results depend on the floating-point implementation (see the chaos band), so exact reproduction needs the same NumPy version.

| Script | Paper section |
|---|---|
| `deney_274_temiz_tur.py`, `deney_274b_bit_genisligi.py`, `deney_274c_kaos_kontrolu.py` | Main comparison, bit-width phase transition, chaos control |
| `deney_275_tek_ray.py` | Signed rail vs dual rail |
| `deney_276_stokastik_kafes.py` | Stochastic lattice step |
| `deney_277_kaos_kontrolu.py` | Autopsy of a false victory |
| `deney_278_float_master_vs_tamsayi.py` | Learning in integers vs a float master |
| `deney_279_pnorm_kafes.py` | p-norm potential on the lattice |
| `deney_280_doygunluk_teshisi.py` | Saturation diagnosis |
| `deney_281_artik_biriktirici.py` | **Integer residual accumulator: 7 bits ≡ 65 bits** |
| `deney_282_yasalarin_sinavi.py` | Bit-width law: T-form refuted, U-form |
| `deney_283_coklu_veri.py`, `deney_283b_dogrulama.py` | Five real datasets |
| `deney_284_azalan_program.py`, `deney_285_adil_dovus.py` | Per-coordinate step schedule, fair baselines |
| `deney_286` … `deney_290` | MNIST chase and the chaos band |
| `deney_291` … `deney_293` | Closing six open gaps |

Code comments and printed labels are in Turkish. A few you will see: *deney* = experiment, *tohum* = seed, *kafes* = lattice, *ham çıktı* = raw output, *artık biriktirici* = residual accumulator.

## License

Free for personal, research and other non-commercial use under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/). For commercial use, contact okmen1306@gmail.com. See [LICENSE](LICENSE).

## Citation

```bibtex
@techreport{okmen2026kafesmwu,
  author      = {Ökmen, Muhammed},
  title       = {Lattice-Quantized Multiplicative Weights: How Many Bits Does an Online Linear Learner Need?},
  institution = {Zenodo},
  year        = {2026},
  doi         = {10.5281/zenodo.23568755}
}
```
