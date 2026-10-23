#!/usr/bin/env python3
"""
01 - Neyman confidence belts and coverage, checked with toys.

  (a) Gaussian estimator: build the belt horizontally, read it vertically.
  (b) Poisson count: the exact (Garwood) central belt; n_obs = 9 gives the
      book's 9 (+6.7, -4.3) at 90% CL, and n_obs = 0 gives the "3 events" UL.
  (c) Coverage vs the true mean: exact Poisson intervals over-cover (discrete n),
      the naive n +- sqrt(n) interval under-covers at small n. Toys reproduce
      the exact sums.
  (d) Confidence regions in 2D: Delta chi2 = 1 covers 39.3%, not 68.3%.

Sources: book confidenceIntervals c2 (Neyman belt, Poisson, n_obs = 0);
2024/Lecture_9-2 p.5-17 and p.21-22; Cowan 9.2-9.4, 9.7 (PDF p.132-145).

usage: python3 01_neyman_coverage.py [--ntoys 20000] [--seed 1] [--outdir plots]
"""
import argparse
import os

import numpy as np
from scipy import stats

import cltoys as ct

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ntoys", type=int, default=20000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--outdir", default="plots")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
rng = np.random.default_rng(args.seed)
plt = ct.setup_style()
C = ct.COLORS

# ---------------------------------------------------------------- (a) Gaussian belt
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axs[0]
theta = np.linspace(0, 10, 201)
z = ct.Phi_inv(1 - (1 - 0.6827) / 2)
ax.fill_betweenx(theta, theta - z, theta + z, color=C[0], alpha=0.25, lw=0,
                 label="acceptance region (68.3%) at each true $\\theta$")
x_obs = 4.2
lo, up = ct.gaussian_interval(x_obs, 1.0, 0.6827)
ax.axvline(x_obs, color="k", ls="--", lw=1.2, label=f"observed $\\hat\\theta$ = {x_obs}")
ax.plot([x_obs, x_obs], [lo, up], color=C[1], lw=5, solid_capstyle="butt",
        label=f"confidence interval [{lo:.2f}, {up:.2f}]")
ax.set_xlabel("estimator $\\hat\\theta$ (measured)")
ax.set_ylabel("true parameter $\\theta$")
ax.set_title("(a) Neyman belt, Gaussian $\\hat\\theta\\sim N(\\theta, 1)$")
ax.legend(loc="upper left", fontsize=8.5)

# ---------------------------------------------------------------- (b) Poisson belt
ax = axs[1]
lam = np.linspace(0.001, 25, 2500)
cl = 0.90
a = (1 - cl) / 2
n_lo, n_hi = [], []
for L in lam:
    n = np.arange(0, int(L + 10 * np.sqrt(L) + 15))
    cdf = stats.poisson.cdf(n, L)
    sf = stats.poisson.sf(n - 1, L)        # P(N >= n)
    n_hi.append(n[cdf >= 1 - a].min())     # largest n kept so that P(N > n2) <= a
    n_lo.append(n[sf >= 1 - a].max())      # smallest n kept so that P(N < n1) <= a
n_lo, n_hi = np.array(n_lo), np.array(n_hi)
ax.fill_betweenx(lam, n_lo - 0.5, n_hi + 0.5, step="pre", color=C[0], alpha=0.25, lw=0,
                 label="90% central acceptance $[n_1(\\lambda), n_2(\\lambda)]$")
nobs = 9
lo9, up9 = ct.poisson_interval(nobs, cl, "central")
ax.axvline(nobs, color="k", ls="--", lw=1.2, label=f"$n_{{obs}}$ = {nobs}")
ax.plot([nobs, nobs], [lo9, up9], color=C[1], lw=5, solid_capstyle="butt",
        label=f"$\\lambda\\in$[{lo9:.2f}, {up9:.2f}] = $9^{{+{up9-9:.1f}}}_{{-{9-lo9:.1f}}}$")
ax.set_xlim(-1, 25)
ax.set_ylim(0, 25)
ax.set_xlabel("observed count $n$")
ax.set_ylabel("true mean $\\lambda$")
ax.set_title("(b) Poisson belt (exact, Garwood), 90% CL")
ax.legend(loc="lower right", fontsize=8.5)
fig.savefig(f"{args.outdir}/01a_neyman_belts.png")

print("== Neyman intervals ==")
print(f"Gaussian x = {x_obs}, sigma = 1, 68.3% CL : [{lo:.3f}, {up:.3f}]")
print(f"Poisson n = 9, 90% central             : [{lo9:.3f}, {up9:.3f}]  (book c2: 9 +6.7 -4.3)")
for c in (0.6827, 0.90, 0.95):
    print(f"Poisson n = 0, {c:.4f} upper limit       : {ct.poisson_interval(0, c, 'upper')[1]:.3f}")
print("  (68.27% two-sided convention for an empty bin: beta = 0.3173/2 ->",
      f"{-np.log(0.3173 / 2):.3f}, the 1.8 error bar of book c2)")

# ---------------------------------------------------------------- (c) coverage
lam_true = np.linspace(0.05, 15, 600)
garwood = lambda n: ct.poisson_interval(n, 0.6827, "central")
naive = lambda n: (n - np.sqrt(n), n + np.sqrt(n))
cov_g = ct.poisson_exact_coverage(lam_true, garwood)
cov_n = ct.poisson_exact_coverage(lam_true, naive)

lam_toy = np.array([0.5, 1.5, 3.0, 5.0, 8.0, 12.0])
toy_g, toy_n = [], []
for L in lam_toy:
    n = rng.poisson(L, args.ntoys).astype(float)
    lo_, up_ = garwood(n)
    toy_g.append(np.mean((lo_ <= L) & (L <= up_)))
    lo_, up_ = naive(n)
    toy_n.append(np.mean((lo_ <= L) & (L <= up_)))
err = np.sqrt(0.68 * 0.32 / args.ntoys)

fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(lam_true, cov_g, color=C[0], label="exact central interval (Neyman/Garwood)")
ax.plot(lam_true, cov_n, color=C[1], ls="--", label="naive $n\\pm\\sqrt{n}$")
ax.errorbar(lam_toy, toy_g, err, fmt="o", ms=6, color=C[0], mfc="white", label=f"toys ({args.ntoys}/point)")
ax.errorbar(lam_toy, toy_n, err, fmt="s", ms=6, color=C[1], mfc="white")
ax.axhline(0.6827, color="k", lw=1, ls=":", label="nominal 68.27%")
ax.set_ylim(0.3, 1.02)
ax.set_xlabel("true Poisson mean $\\lambda$")
ax.set_ylabel("coverage  $P(\\lambda \\in [\\lambda_1(n), \\lambda_2(n)])$")
ax.set_title("(c) Coverage of 68.3% Poisson intervals")
ax.legend(loc="lower right", fontsize=8.5)
fig.savefig(f"{args.outdir}/01b_poisson_coverage.png")

print("\n== Coverage (68.27% nominal) ==")
print(" lambda  exact-Neyman  toys     naive n+-sqrt(n)  toys")
for L, tg, tn in zip(lam_toy, toy_g, toy_n):
    eg = ct.poisson_exact_coverage([L], garwood)[0]
    en = ct.poisson_exact_coverage([L], naive)[0]
    print(f" {L:5.1f}   {eg:.4f}        {tg:.4f}   {en:.4f}            {tn:.4f}")
print(f" min exact coverage over lambda in (0, 15]: Neyman {cov_g.min():.4f}, naive {cov_n.min():.4f}")

# ---------------------------------------------------------------- (d) 2D regions
rho = 0.6
cov = np.array([[1, rho], [rho, 1]])
x = rng.multivariate_normal([0, 0], cov, args.ntoys)        # (hat a, hat b) - truth
vinv = np.linalg.inv(cov)
dchi2 = np.einsum("ti,ij,tj->t", x, vinv, x)
print("\n== 2D confidence regions (Gaussian estimators, rho = 0.6) ==")
for q in (1.0, 2.30, 4.0, 6.18):
    print(f" P(Delta chi2 <= {q:4.2f}) toys = {np.mean(dchi2 <= q):.4f}   exact chi2(2 dof) = {stats.chi2.cdf(q, 2):.4f}")
print(f" 1D interval |a_hat - a| <= 1 alone covers {np.mean(np.abs(x[:, 0]) <= 1):.4f} (68.27%)")
print(" -> for a joint 68.3% region in 2D use Delta chi2 = 2.30 (L9 p.21-22; Cowan 9.7)")

fig, ax = plt.subplots(figsize=(5.2, 5))
ax.plot(x[:3000, 0], x[:3000, 1], ".", ms=2, color="0.6", alpha=0.6, label="toy estimates (first 3000)")
tt = np.linspace(0, 2 * np.pi, 400)
L = np.linalg.cholesky(cov)
for q, c, ls in ((1.0, C[1], "--"), (2.30, C[0], "-")):
    e = (L @ np.vstack([np.cos(tt), np.sin(tt)])) * np.sqrt(q)
    ax.plot(e[0], e[1], color=c, ls=ls,
            label=f"$\\Delta\\chi^2$ = {q}: covers {stats.chi2.cdf(q, 2) * 100:.1f}%")
ax.axvspan(-1, 1, color=C[2], alpha=0.12, label="1D $\\pm1\\sigma$ on $a$: 68.3%")
ax.set_xlabel("$\\hat a - a$")
ax.set_ylabel("$\\hat b - b$")
ax.set_title("(d) 2D regions, $\\rho$ = 0.6")
ax.legend(fontsize=8, loc="upper left")
ax.set_aspect("equal")
fig.savefig(f"{args.outdir}/01c_2d_regions.png")
print(f"\nplots written to {args.outdir}/01a_neyman_belts.png, 01b_poisson_coverage.png, 01c_2d_regions.png")
