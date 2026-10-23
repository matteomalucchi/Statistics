#!/usr/bin/env python3
"""
02 - Limits near a physical boundary: classical, flip-flop, Feldman-Cousins,
     Bayesian. With exact coverage and toy checks.

  (a) Gaussian x ~ N(mu, 1) with mu >= 0 (e.g. a mass, a cross section):
      90% intervals vs x for the classical upper limit, the "flip-flop" policy,
      Feldman-Cousins (FC) and a Bayesian flat-prior upper limit.
  (b) Coverage vs true mu: flip-flop under-covers (~85% for 1.4 < mu < 4.3),
      FC covers by construction.
  (c) Poisson n ~ Pois(mu + b), b = 3: FC belt; FC table check against
      FC paper Table IV; classical interval empty at n = 0.
  (d) Under-fluctuations and sensitivity: a worse experiment with a downward
      fluctuation gets a better limit; report the sensitivity (mean UL of
      background-only toys) next to it (FC paper Sec. VI; book c11).

Sources: book confidenceIntervals c4-c11; 2024/Lecture_13-2 p.5-57;
papers/Feldman-Cousins_PhysRevD.57.3873.pdf (Sec. III-VI, tables p.7-11);
Cowan 9.8 (PDF p.149).

usage: python3 02_boundaries_feldman_cousins.py [--ntoys 20000] [--b 3] [--outdir plots]
"""
import argparse
import os

import numpy as np
from scipy import stats

import cltoys as ct

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ntoys", type=int, default=20000)
ap.add_argument("--seed", type=int, default=2)
ap.add_argument("--b", type=float, default=3.0, help="known background for the Poisson part")
ap.add_argument("--cl", type=float, default=0.90)
ap.add_argument("--outdir", default="plots")
ap.add_argument("--fc-monotone", action="store_true",
                help="apply FC's monotonicity-in-b correction to the n=0 entry (slow)")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
rng = np.random.default_rng(args.seed)
plt = ct.setup_style()
C = ct.COLORS
CL = args.cl

# ======================================================================= Gaussian
print(f"== Gaussian with boundary mu >= 0, sigma = 1, CL = {CL} ==")
belt = ct.fc_gaussian_belt(CL, mu_grid=np.linspace(0, 9, 901))
x = np.linspace(-2.5, 4.5, 281)
ul_classic = x + ct.Phi_inv(CL)
ff_lo, ff_up = ct.flipflop_interval(x, CL)
fc = np.array([ct.fc_gaussian_interval(xi, belt) for xi in x])
# Bayesian, flat prior on mu >= 0:  mu_up = x + Phi^{-1}(1 - (1-CL) Phi(x))
ul_bayes = x + ct.Phi_inv(1 - (1 - CL) * ct.Phi(x))

fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axs[0]
ax.plot(x, ul_classic, color=C[1], ls=":", label="classical UL $x + 1.28$ (can be < 0)")
ax.plot(x, ff_up, color=C[3], ls="--", label="flip-flop (upper/lower ends)")
ax.plot(x, ff_lo, color=C[3], ls="--")
ax.plot(x, fc[:, 1], color=C[0], label="Feldman–Cousins (upper/lower ends)")
ax.plot(x, fc[:, 0], color=C[0])
ax.plot(x, ul_bayes, color=C[2], ls="-.", label="Bayes flat prior UL (= CLs UL)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("measured $x$ (units of $\\sigma$)")
ax.set_ylabel("$\\mu$ interval / limit")
ax.set_title(f"(a) {CL:.0%} intervals for $\\mu\\geq0$")
ax.set_ylim(-1.5, 6.5)
ax.legend(fontsize=8, loc="upper left")

mu_t = np.linspace(0, 6, 241)
cov_ff = ct.gaussian_interval_coverage(mu_t, lambda xx: ct.flipflop_interval(xx, CL))
x_tab = np.linspace(-7.5, 20, 2751)
fc_tab = np.array([ct.fc_gaussian_interval(xi, belt) for xi in x_tab])
# outside the tabulated belt (x > 14) FC = central interval
zc = ct.Phi_inv(1 - (1 - CL) / 2)
fc_lo_tab = np.where(np.isnan(fc_tab[:, 0]), x_tab - zc, fc_tab[:, 0])
fc_up_tab = np.where(np.isnan(fc_tab[:, 1]), x_tab + zc, fc_tab[:, 1])
cov_fc = ct.gaussian_interval_coverage(mu_t, lambda xx: (np.interp(xx, x_tab, fc_lo_tab),
                                                          np.interp(xx, x_tab, fc_up_tab)))
cov_cl = ct.gaussian_interval_coverage(mu_t, lambda xx: (np.full_like(xx, -np.inf), xx + ct.Phi_inv(CL)))

# toy check of the flip-flop under-coverage at mu = 2
mu0 = 2.0
xt = rng.normal(mu0, 1, args.ntoys)
lo_, up_ = ct.flipflop_interval(xt, CL)
toy_ff = np.mean((lo_ <= mu0) & (mu0 <= up_))
lo_, up_ = np.interp(xt, x_tab, fc_lo_tab), np.interp(xt, x_tab, fc_up_tab)
toy_fc = np.mean((lo_ <= mu0) & (mu0 <= up_))
e = np.sqrt(CL * (1 - CL) / args.ntoys)

ax = axs[1]
ax.plot(mu_t, cov_ff, color=C[3], ls="--", label="flip-flop")
ax.plot(mu_t, cov_fc, color=C[0], label="Feldman–Cousins")
ax.plot(mu_t, cov_cl, color=C[1], ls=":", label="classical UL")
ax.errorbar([mu0, mu0], [toy_ff, toy_fc], [e, e], fmt="o", color="k", mfc="white",
            label=f"toys at $\\mu$ = {mu0} ({args.ntoys})")
ax.axhline(CL, color="k", lw=0.8, ls=":")
ax.set_ylim(0.8, 1.005)
ax.set_xlabel("true $\\mu$")
ax.set_ylabel("coverage")
ax.set_title("(b) coverage of the Gaussian prescriptions")
ax.legend(fontsize=8, loc="lower right")
fig.savefig(f"{args.outdir}/02a_gaussian_boundary.png")

bad = mu_t[cov_ff < CL - 0.005]
print(f" flip-flop coverage minimum {cov_ff.min():.4f}; below {CL - 0.005:.3f} for mu in "
      f"[{bad.min():.2f}, {bad.max():.2f}]  (FC paper: 85% for 1.36 < mu < 4.28)")
print(f" FC coverage range [{cov_fc.min():.4f}, {cov_fc.max():.4f}]")
print(f" toys at mu = {mu0}: flip-flop {toy_ff:.4f} +- {e:.4f}, FC {toy_fc:.4f} +- {e:.4f}")
for xo in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
    l, u = ct.fc_gaussian_interval(xo, belt)
    print(f" x = {xo:+.1f}: classical UL {xo + ct.Phi_inv(CL):+.3f} | FC [{l:.2f}, {u:.2f}] | "
          f"Bayes/CLs UL {xo + ct.Phi_inv(1 - (1 - CL) * ct.Phi(xo)):.3f}")
print("  (FC paper Table X p.10, 90%: x=-2 [0,0.40]; -1 [0,0.81]; 0 [0,1.64]; 1 [0,2.64]; 2 [0.58,3.64]; 3 [1.37,4.64])")

# ======================================================================= Poisson
b = args.b
print(f"\n== Poisson n ~ Pois(mu + b), b = {b}, CL = {CL} ==")
mu_grid = np.arange(0, 20.0001, 0.005)
pbelt = ct.fc_poisson_belt(b, CL, mu_grid)
print("  n | classical central            | classical UL | FC interval")
rows = []
for n in range(0, 16):
    clo, cup = ct.poisson_interval(n, CL, "central", b)
    ul = ct.poisson_interval(n, CL, "upper", b)[1]
    lo, up = ct.fc_poisson_interval(n, b, CL, belt=pbelt)
    rows.append((n, lo, up))
    cen = "EMPTY" if cup < 0 else f"[{max(clo, 0):5.2f}, {cup:5.2f}]"
    uls = "EMPTY" if ul < 0 else f"{ul:5.2f}"
    print(f" {n:2d} | {cen:28s} | {uls:12s} | [{lo:5.2f}, {up:5.2f}]")
print("  FC paper Table IV (b = 3): n=0 [0,1.08], 1 [0,1.88], 2 [0,3.04], 3 [0,4.42], 4 [0,5.60],\n"
      "  5 [0,6.99], 6 [0.15,8.47], 7 [0.89,9.53], 8 [1.51,10.99], 9 [1.88,12.30], 10 [2.63,13.50]")
if b == 3.0:
    if args.fc_monotone:
        lo0, up0 = ct.fc_poisson_interval(0, b, CL, monotone_b=True)
        print(f"  with the monotone-in-b correction: n = 0 -> [{lo0:.2f}, {up0:.2f}]")
    else:
        print("  (n = 0: the raw construction gives 0.95; FC lengthen it to 1.08 so that mu_up\n"
              "   is non-increasing in b - run with --fc-monotone to reproduce)")
lo, hi, p = ct.fc_poisson_acceptance(0.5, b, CL)
print(f"  acceptance region for mu = 0.5: n in [{lo}, {hi}], P = {p:.3f}  (book c7: 0.935)")

fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axs[0]
mu, nlo, nhi = pbelt
ax.fill_betweenx(mu, nlo - 0.5, nhi + 0.5, step="pre", color=C[0], alpha=0.25, lw=0,
                 label="FC acceptance region")
lam = mu + b
cl_lo = np.array([np.arange(200)[stats.poisson.sf(np.arange(200) - 1, L) >= 1 - (1 - CL) / 2].max() for L in lam])
cl_hi = np.array([np.arange(200)[stats.poisson.cdf(np.arange(200), L) >= 1 - (1 - CL) / 2].min() for L in lam])
ax.step(cl_lo - 0.5, mu, color=C[1], ls="--", lw=1.4, label="classical central belt edges")
ax.step(cl_hi + 0.5, mu, color=C[1], ls="--", lw=1.4)
ns = np.array([r[0] for r in rows])
ax.vlines(ns, [r[1] for r in rows], [r[2] for r in rows], color=C[0], lw=3)
ax.set_xlim(-0.8, 15.5)
ax.set_ylim(0, 15)
ax.set_xlabel("observed $n$")
ax.set_ylabel("signal mean $\\mu$")
ax.set_title(f"(c) FC belt, Poisson with b = {b:g}, {CL:.0%} CL")
ax.legend(fontsize=8, loc="upper left")

# coverage of FC (exact), plus classical UL
mut = np.linspace(0, 10, 401)
nmax = 60
fc_lo_n = np.array([r[1] for r in rows] + [ct.fc_poisson_interval(n, b, CL, belt=pbelt)[0] for n in range(16, nmax)])
fc_up_n = np.array([r[2] for r in rows] + [ct.fc_poisson_interval(n, b, CL, belt=pbelt)[1] for n in range(16, nmax)])
ncount = np.arange(nmax)
cov_fcp = [stats.poisson.pmf(ncount[(fc_lo_n <= m) & (m <= fc_up_n)], m + b).sum() for m in mut]
ul_n = ct.poisson_interval(ncount, CL, "upper", b)[1]
cov_ul = [stats.poisson.pmf(ncount[m <= ul_n], m + b).sum() for m in mut]
ax = axs[1]
ax.plot(mut, cov_fcp, color=C[0], label="Feldman–Cousins")
ax.plot(mut, cov_ul, color=C[1], ls=":", label="classical upper limit")
ax.axhline(CL, color="k", lw=0.8, ls=":")
ax.set_ylim(0.85, 1.005)
ax.set_xlabel("true signal mean $\\mu$")
ax.set_ylabel("coverage")
ax.set_title("(d) exact coverage, Poisson with background")
ax.legend(fontsize=8, loc="lower right")
fig.savefig(f"{args.outdir}/02b_poisson_fc.png")
print(f"  FC exact coverage over mu in [0, 10]: min {min(cov_fcp):.4f}")

# ======================================================================= sensitivity
print(f"\n== Under-fluctuations and sensitivity ({CL:.0%} CL) ==")
fc00 = ct.fc_poisson_interval(0, 0.0, CL)[1]
fc105 = ct.fc_poisson_interval(5, 10.0, CL, mu_grid=np.arange(0, 15, 0.005))[1]
print(f" b = 0,  n = 0: FC UL = {fc00:.2f};  CLs UL = {ct.counting_upper_limit(0, 0.0, CL, 'CLs'):.2f}")
print(f" b = 10, n = 5: FC UL = {fc105:.2f};  CLs UL = {ct.counting_upper_limit(5, 10.0, CL, 'CLs'):.2f}")
print("  -> the worse experiment (b = 10) with a lucky deficit quotes the better FC limit (book c11).")
bs = np.array([0.0, 1.0, 3.0, 5.0, 10.0])
for bb in bs:
    belt_b = ct.fc_poisson_belt(bb, CL, np.arange(0, 25, 0.01))
    nmax_b = int(bb + 8 * np.sqrt(bb + 1) + 10)
    ul_fc = np.array([ct.fc_poisson_interval(n, bb, CL, belt=belt_b)[1] for n in range(nmax_b)])
    ul_cls = np.array([ct.counting_upper_limit(n, bb, CL, "CLs") for n in range(nmax_b)])
    w = stats.poisson.pmf(np.arange(nmax_b), bb)
    n_toy = rng.poisson(bb, args.ntoys)
    n_toy = np.minimum(n_toy, nmax_b - 1)
    print(f" b = {bb:4.1f}: sensitivity = mean UL of b-only toys: FC {ul_fc[n_toy].mean():.2f} "
          f"(exact {np.sum(w * ul_fc):.2f}), CLs {ul_cls[n_toy].mean():.2f}; median CLs {np.median(ul_cls[n_toy]):.2f}")
print("  FC paper Table XII (p.16), 90% CL sensitivity: b=0 2.44, b=1 3.28, b=3 4.42, b=5 5.18, b=10 6.76\n"
      "  (ours is ~0.01-0.03 lower: no monotone-in-b correction)")
print(f"\nplots written to {args.outdir}/02a_gaussian_boundary.png and 02b_poisson_fc.png")
