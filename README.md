# Monte Carlo Reproduction: Marginal-Transform Inertness for Local Gaussian Correlation Estimation

This repository implements eight Monte Carlo experiments studying **Local Gaussian Correlation (LGC)** estimation, focusing on:

1. **Marginal-transform inertness** — whether replacing the canonical empirical normal-score transform with oracle or fitted heavy-tailed transforms materially changes LGC accuracy.
2. **Adaptive bandwidth performance** — the corrected AMISE-based adaptive bandwidth rule compared to the global plug-in across dependence regimes.
3. **Real-data application** — bootstrap stability of LGC surfaces on ETF return pairs (SPY/TLT, SPY/EEM) after GARCH volatility filtering.

---

## Repository Structure

```
.
├── data/                 # Data loading scripts (financial data via Massive API)
│   └── download.py
├── src/                  # Core implementation
│   ├── nts.py            # Normal-Inverse-Gaussian (NIG/NTS) marginal distribution
│   ├── copulas.py        # Copula simulation and τ-to-parameter mappings
│   ├── transforms.py     # Gaussianisation transforms (empirical, oracle, fitted)
│   ├── lgc.py            # Local Gaussian Correlation (LGC) estimator
│   ├── bandwidth.py      # Global plug-in and corrected adaptive bandwidth rules
│   └── metrics.py        # Density-weighted ISE, effective sample size, diagnostics
├── exp/                  # Experiment scripts
│   ├── exp_1.py          # Symmetric NTS inertness (Exp 1)
│   ├── exp_2.py          # Skewed NTS robustness (Exp 2)
│   ├── exp_3.py          # Scarcity diagnostics (Exp 3)
│   ├── exp_4.py          # Bias-expansion validation (Exp 4)
│   ├── exp_5.py          # Adaptive vs global at moderate dependence (Exp 5)
│   ├── exp_6.py          # Dependence-regime map (Exp 6)
│   ├── exp_7.py          # Strong-dependence failure diagnostics (Exp 7)
│   └── exp_8.py          # Real-data bootstrap stability (Exp 8)
├── tests/                # pytest test suite (236 tests)
├── results/              # Output figures, CSV tables, and RESULTS.md
│   └── RESULTS.md
├── AGENTS.md             # Repository skill file
├── requirements.txt
└── README.md
```

---

## Methodology

### Copula Families
- **Gaussian** — standard bivariate normal copula
- **Clayton** — lower-tail-dependent Archimedean copula
- **t(ν=4)** — symmetric tail-dependent elliptical copula
- **Gumbel** — upper-tail-dependent Archimedean copula

All copulas are parameterised at Kendall's τ = 0.5 (main experiments) and τ ∈ {0.2, 0.5, 0.8} (regime map).

### Marginal Distributions
Heavy-tailed marginals are approximated via **Normal-Inverse-Gaussian (NIG)** distributions (tractable proxy for Normal Tempered Stable / NTS). The symmetric variant uses δ=1, μ=0, β=0 scaled to unit variance. The skewed variant uses β=−0.6 (Exp 2).

### LGC Estimator
The LGC estimator maximises a local bivariate Gaussian log-likelihood weighted by a product Gaussian kernel:

```
L_n(θ; x) = (1/n) Σ_i K_b(Z_i − x) log ψ(Z_i; θ) − ∫ K_b(u − x) ψ(u; θ) du
```

where θ = (μ₁, μ₂, σ₁, σ₂, ρ), K_b is the scaled product kernel, and Z_i are Gaussianised observations. The local correlation ρ̂(x) is extracted from the fitted θ.

### Bandwidth Rules
- **Global plug-in:** b₀ = 1.75 · n^(−1/6)
- **Corrected AMISE adaptive:** pointwise AMISE-optimal bandwidth using pilot-estimated ρ̃, density f̃, and bias functional β̃ = Δρ + 2(∇ log f)ᵀ∇ρ, with budget-neutral normalisation and cap at [0.5, 3] · b₀

### Gaussianisation Transforms
1. **Empirical:** rank-based CDF followed by Φ⁻¹ (convention: (rank − 0.5) / n)
2. **Oracle:** true NIG CDF followed by Φ⁻¹
3. **Fitted:** maximum-likelihood NIG followed by Φ⁻¹

---

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run All Tests
```bash
python -m pytest tests/ -q
```

### Run Individual Experiments
```bash
python exp/exp_1.py   # Marginal-transform inertness
python exp/exp_2.py   # Skewed NTS robustness
python exp/exp_3.py   # Scarcity diagnostics
python exp/exp_4.py   # Bias-expansion validation
python exp/exp_5.py   # Adaptive vs global (moderate dependence)
python exp/exp_6.py   # Dependence-regime map
python exp/exp_7.py   # Strong-dependence failure diagnostics
python exp/exp_8.py   # Real-data bootstrap stability
```

### Financial Data
Exp 8 uses real ETF data downloaded via the Massive API. Set your API key:
```bash
export MASSIVE_TOKEN=your_api_key_here
python data/download.py
```

---

## Key Results

See [`results/RESULTS.md`](results/RESULTS.md) for full numeric tables and discussion.

**Core finding — Exp 1 & 2 (confirmed):** The canonical empirical normal-score transform achieves essentially the same density-weighted ISE as oracle and fitted heavy-tailed transforms. Differences appear only in the 3rd–4th decimal place, confirming marginal-transform inertness for LGC estimation.

**Core finding — Exp 3 (partial):** Local effective sample size eff_n(x) is the primary organiser of LGC estimation error. Fisher-normalised SD regression on log(eff_n) yields slope −0.51 (paper: −0.61) and R² = 0.43 (paper: 0.76).

**Core finding — Exp 7 (confirmed):** Under strong dependence, even an oracle scalar multiplier over the adaptive AMISE shape cannot rescue performance relative to the global plug-in, confirming that strong-dependence failure is structural.

**GARCH diagnostics — Exp 8 (confirmed):** AR(1)-GARCH(1,1)-t fits for SPY and TLT match the paper's Table 3 estimates to within rounding: SPY (α₁=0.141, β₁=0.854, ν=5.55) and TLT (α₁=0.054, β₁=0.941, ν=15.5).

---

## Dependencies

- Python ≥ 3.10
- numpy, scipy, pandas, statsmodels, matplotlib, seaborn
- arch (GARCH estimation)
- pyvinecopulib (copula simulation)
- pytest (testing)
