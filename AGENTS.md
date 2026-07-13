# LGC Monte Carlo -- Agent Memory

## Project Layout

```
src/          Source modules (do not modify without good reason)
exp/          Experiment scripts (exp_1.py ... exp_8.py)
results/      Output CSVs and PNGs (created at runtime via os.makedirs)
tests/        Test suite
```

## Source Module Quick Reference

| Module | Key exports |
|--------|-------------|
| `src/nts.py` | `NIGDistribution`, `symmetric_nts()`, `skewed_nts()` |
| `src/copulas.py` | `ALL_FAMILIES`, `make_bicop(family, tau)`, `simulate_copula_with_margins()` |
| `src/lgc.py` | `lgc_surface()`, `global_bandwidth()`, `make_eval_grid()`, `density_weighted_ise()` |
| `src/transforms.py` | `gaussianize()`, transform_type in ['empirical','oracle','fitted'] |
| `src/simulation.py` | `run_mc_cell()`, `build_truth_surface()` |
| `src/bandwidth.py` | `corrected_adaptive_bandwidth()`, `heuristic_adaptive_bandwidth()`, `lgc_surface_adaptive()` |
| `src/metrics.py` | `aggregate_ise()`, `pointwise_diagnostics()` |

`ALL_FAMILIES = ['gaussian', 'clayton', 't4', 'gumbel']` (first 2 used in FAST_MODE).

`make_eval_grid()` returns a `(126, 2)` grid of evaluation points.

## src/nts.py -- Bugs Fixed (do not revert)

Two pre-existing bugs were corrected; do not revert these changes:

### 1. `_nig_raw_skewness` wrong formula

The original formula `3.0 * b * gamma**1.5 / (a**2 * gamma**0.5)` simplified to
`3*b*gamma/a^2`, which is bounded and gave same-sign at both brentq endpoints,
causing `_calibrate_skewed_params` (and thus `skewed_nts()`) to raise ValueError.

Correct formula (matches `scipy.stats.norminvgauss.stats(moments='s')`):

    3*b / (a * (a^2 - b^2)^{1/4})

Code: `return 3.0 * b / (a * gamma**0.5) if a > abs(b) else np.nan`

### 2. `NIGDistribution.ppf` / `.cdf` too slow

`scipy.stats.norminvgauss` falls back to element-wise numerical root-finding in
scipy <= 1.14 (~33 ms/point), making `n_gt=20000` take >10 min per truth-surface.

Fix: `_build_lookup_table()` integrates the fast vectorised PDF on a 20 000-point
grid using `scipy.integrate.cumulative_trapezoid`, then `ppf`/`cdf` use `np.interp`.

- Table built once in `__init__` (~8 ms total for two NIGDistribution objects).
- Subsequent calls: <2 ms for 20 000 points.
- Round-trip accuracy: max|cdf(ppf(q)) - q| < 2e-17 (machine precision).

## Critical Performance Note -- scipy norminvgauss ppf/cdf

`scipy.stats.norminvgauss` does NOT have an analytical CDF/PPF in scipy <= 1.14.
`NIGDistribution` now provides fast equivalents via a lookup table (see above).
Always use `NIGDistribution.ppf` / `.cdf` -- never call `self._dist.ppf/cdf` directly.

Benchmarks (Gaussian copula, tau=0.5):
- `NIGDistribution.ppf(20000 pts)` -> ~2 ms  (after lookup-table fix)
- `cop.simulate(20000)` + `empirical_transform` + `lgc_surface` -> ~5 s (truth surface)
- Full replication (simulate + transform + lgc_surface) for n=500 -> <0.5 s
- `corrected_adaptive_bandwidth` n=500 -> <0.01 s
- `heuristic_adaptive_bandwidth` n=2500 -> ~0.06 s
- `norminvgauss._dist.ppf(500 pts)` raw -> ~18 s  <-- NEVER call this directly

## Experiment Conventions

- `FAST_MODE = True` at top of each script for rapid smoke-testing.
- `BASE_SEED` differs per experiment (42 for exp_1/exp_2, 5000 for exp_5, 6000 for exp_6).
- Truth surface is built once per `(family, tau)` cell, reused across all n and reps.
- Paired design: same `Z` sample feeds all estimators within one replication.
- Outputs go to `results/exp_N_*.{csv,png}`.

## exp_1.py (Marginal-Transform Inertness)

- **Purpose**: Reproduce marginal-transform inertness for LGC with symmetric NIG marginals.
- **FAST_MODE**: copulas=['gaussian','clayton'], n={500,1000}, R=3, transforms=['empirical','oracle']
- **Full mode**: 4 copulas, n={500,1000,2500,5000}, R=50, transforms=['empirical','oracle','fitted']
- `tau=0.5`, `n_gt=20000`, `base_seed=42`
- Outputs: `exp_1_raw.csv`, `exp_1_summary.csv`, `exp_1_ise_table.png` (table-style heatmap)
- ISE aggregation: groupby ['family','tau','n','transform'], report mean_ise and sd_ise.

## exp_2.py (Asymmetric Marginal Robustness)

- **Purpose**: Same structure as exp_1 but with skewed NIG (target_skew=-0.84), fixed n=2500.
- **FAST_MODE**: copulas=['gaussian','clayton'], R=3
- **Full mode**: 4 copulas, R=25, transforms=['empirical','oracle','fitted']
- `tau=0.5`, `n_gt=20000`, `base_seed=42`
- Outputs: `exp_2_raw.csv`, `exp_2_summary.csv`, `exp_2_ise_comparison.png` (bar chart)

## exp_3.py (Scarcity Diagnostic for Tail LGC Estimation Error)

- **Purpose**: Diagnose how LGC estimation error (Fisher-normalized SD) scales with
  local effective sample size (eff_n) via pooled log-log regression.
- **Settings**: symmetric NTS, tau=0.5, empirical transform only
- **FAST_MODE**: copulas=['gaussian','clayton'], n={500,1000}, R=3
- **Full mode**: 4 copulas, n={500,1000,2500,5000}, R=50
- `n_gt=20000`, `BASE_SEED=42`
- Uses `run_mc_cell()` then pivots per-rep DataFrame to `(R, M)` arrays via
  `pivot_table(index='replication', columns='point_idx', values=..., aggfunc='first')`.
- Calls `pointwise_diagnostics(rho_hat_reps, rho_gt_arr, eff_n_reps)` per cell.
- Pools all cells, fits `scarcity_regression(sd, rho_gt, mean_eff_n)` on pooled data.
- Expected slope ≈ −0.5 (Fisher information theory prediction).
- Outputs: `exp_3_diagnostics.csv` (per-point), `exp_3_regression.csv` (slope/R²),
  `exp_3_scarcity_plot.png` (scatter log(eff_n) vs log(Fisher SD) + OLS line).

## exp_4.py (Bias-Expansion Validation for LGC Estimator)

- **Purpose**: Validate the second-order bias expansion E[rho_hat(x;b)] ≈ rho(x) + beta(x)*b²
  by sweeping a bandwidth ladder and regressing mean estimates on b² per grid point.
- **Settings**: Clayton and Gumbel only, n=4000, symmetric NTS, empirical transform
- **Bandwidth ladder**: `c_j * n^(-1/6)` where `c_j ∈ [0.6, 0.8, ..., 2.4]`
- **FAST_MODE**: copulas=['clayton'], R=10, 2 bandwidths (c_j=0.6, 0.8)
- **Full mode**: copulas=['clayton','gumbel'], R=120, 10 bandwidths
- `n_gt=20000`, `BASE_SEED=4000`
- Replications are **paired across bandwidths**: rep r always uses seed `BASE_SEED + r*100`,
  so samples are the same for all bandwidths within each rep.
- `_compute_beta_tilde(family, tau, marginal, grid, n_gt, seed)`: generates a fresh large
  pilot sample → empirical Gaussianize → `lgc_surface` at `global_bandwidth(n_gt)` →
  `pilot_density` → `estimate_beta_field(rho_pilot, f_pilot, grid, n_side=11)`.
- Per-point regression: `scipy.stats.linregress(b_sq[valid], mean_rho[valid])` per grid point.
- `gamma` = through-origin OLS: `dot(c, beta_tilde) / dot(beta_tilde, beta_tilde)`.
- Outputs: `exp_4_bias_expansion.csv` (per-point slope_c, intercept_a, r2, beta_tilde, gamma),
  `exp_4_slope_vs_beta.png` (scatter c(x) vs beta_tilde(x) + through-origin line per family).

## exp_5.py (Adaptive vs Global vs Heuristic, tau=0.5)

- **FAST_MODE**: 2 families, n={500,1000}, R=2, estimators=[global, corrected]
- **Full mode**: 4 families, n={500,1000,2500,5000}, R=50, estimators=[global, corrected, heuristic]
- Outputs: `exp_5_raw.csv`, `exp_5_summary.csv`, `exp_5_comparison.png`

## exp_6.py (Dependence-Regime Map)

- **FAST_MODE**: 2 families, tau={0.2,0.5}, n={500,1000}, R=2 for each tau
- **Full mode**: 4 families, tau={0.2,0.5,0.8}, n={500,1000,2500,5000}, R={25,50,25}
- Outputs: `exp_6_raw.csv`, `exp_6_regime_map.csv`, `exp_6_heatmap.png`
- Heatmap color: RdYlGn_r -- green = adaptive better (negative % change), red = adaptive worse

## exp_7.py (Strong-Dependence Failure Diagnostics for Adaptive LGC)

- **Purpose**: Diagnose where corrected adaptive bandwidth fails at tau=0.8 via cap-hit frequency,
  global vs adaptive ISE, and an oracle-scale sweep finding optimal c* for each copula.
- **FAST_MODE**: copulas=['clayton','t4'], n={500,2500}, R=2, c_step=0.5
- **Full mode**: copulas=['clayton','t4','gumbel'], n={500,1000,2500,5000}, R=25, c_step=0.1
- tau=0.8, N_SWEEP=2500, N_GT=20000, BASE_SEED=42
- Part 1: global vs adaptive ISE + cap-hit frequency at each (copula, n)
- Part 2: oracle sweep b_c(x) = clip(c * s(x), 0.5*b0, 3*b0) over c in [0.3,3.0)
- Outputs: `exp_7_diagnostics.csv`, `exp_7_oracle_sweep.csv`,
  `exp_7_oracle_sweep.png`, `exp_7_ise_comparison.png`

**Key design choice**: exp_7 uses `_simulate_gaussianize()` (copula uniforms + empirical
transform) instead of `simulate_copula_with_margins()` + `NIGDistribution.ppf`. Empirical
Gaussianization (Φ⁻¹((rank-0.5)/n)) is rank-invariant, so the LGC surfaces are identical
whether you apply it to copula uniforms U or to F⁻¹(U) for any strictly-increasing F.
This bypasses the slow NIG ppf entirely and keeps each rep well under 1 s.

`compute_adaptive_shape(Z, grid)` replicates the shape_raw internal to
`corrected_adaptive_bandwidth()` before normalization; used by the oracle sweep.

## exp_8.py (Real-Data Bootstrap Stability)

- **Purpose**: Bootstrap stability of global vs adaptive LGC on filtered ETF returns
  (SPY/TLT, SPY/EEM). If download fails, uses synthetic bivariate normal residuals.
- **FAST_MODE**: PAIRS=[('SPY','TLT')], B=10
- **Full mode**: PAIRS=[('SPY','TLT'),('SPY','EEM')], B=200
- GARCH: `arch_model(..., mean='AR', lags=1, vol='GARCH', p=1, q=1, dist='t')`
- Ljung-Box on squared std_resid: `acorr_ljungbox(std_resid**2, lags=[10])`
- Returns scaled ×100 for ARCH numerical stability; std_resid is scale-invariant
- Fallback: `np.random.multivariate_normal([0,0], [[1,0.3],[0.3,1]], 1000)`, seed=42
- GARCH diagnostics show NaN for synthetic-data fallback (no real model fitted)
- Outputs: `exp_8_garch_diagnostics.csv`, `exp_8_bootstrap_stability.csv`,
  `exp_8_lgc_surfaces.png` (pcolormesh heatmaps), `exp_8_sd_ratio.png` (scatter)
- Bootstrap SD ratio > 1 → adaptive less stable; Share < 0.5 typical at strong dependence

## Rank-Invariance of Empirical Gaussianization

For all experiments using `transform_type='empirical'`: the mapping
Z_i = Φ⁻¹((rank(X_i) − 0.5)/n) depends only on the ranks of X, not on the marginal
distribution. Since rank(F⁻¹(U_i)) = rank(U_i) for any strictly-increasing F, applying
the empirical transform to copula uniform draws U gives identical Z to applying it to
F⁻¹(U) for NIG or any other marginal. This means `simulate_copula_with_margins` +
empirical transform = copula simulation alone + empirical transform — use the faster path.
