#!/usr/bin/env python3
"""
03 - Exclusion limits for a counting experiment: CL_{s+b}, CLs, Bayesian,
     expected limits from background-only toys, and the LHC test statistic
     with a log-normal background uncertainty (the Combine "CLs by hand" example).

  (a) CL_{s+b}(s), CL_b and CLs(s) for an observed count; the 95% CL limits.
  (b) Table of upper limits vs n_obs: classical CL_{s+b}, CLs and Bayesian with
      a flat prior; CLs and flat-prior Bayes coincide exactly for this problem.
  (c) Why CLs: probability to exclude a signal s when only background is
      present. CL_{s+b} excludes even s -> 0 in 5% of the experiments
      ("spurious exclusion"); CLs does not. Price: over-coverage.
  (d) Expected limit and +-1/2 sigma bands from background-only toys, compared
      with the asymptotic (Asimov) formula, for small and large background.
  (e) With a nuisance: N_obs = 4, B = 3.26, kappa = 1.2353 (log-normal on b),
      q~_mu with frequentist toys vs asymptotic CLs (Combine
      data/benchmarks/simple-counting/lhcClsByHand.cxx).

Sources: book confidenceIntervals c14 (CLs), c26 (Bayesian UL), c20 (expected
bands); 2024/Lecture_9-2 p.39-46; 2024/STAMET_May14_2024 p.34-43;
C:what_combine_does/statistical_tests L173; papers/1007.1727.pdf Sec. 3.7, 4.3.

usage: python3 03_cls_counting.py [--b 3] [--nobs 2] [--ntoys 20000] [--outdir plots]
"""
import argparse
import os

import numpy as np
from scipy import stats

import cltoys as ct

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--b", type=float, default=3.0)
ap.add_argument("--nobs", type=int, default=2)
ap.add_argument("--cl", type=float, default=0.95)
ap.add_argument("--ntoys", type=int, default=20000)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--outdir", default="plots")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
rng = np.random.default_rng(args.seed)
plt = ct.setup_style()
C = ct.COLORS
b, nobs, CL = args.b, args.nobs, args.cl
alpha = 1 - CL

# ---------------------------------------------------------------- (a) CLs curves
s = np.linspace(0, 12, 601)
clsb, clb, cls = ct.counting_cls(nobs, s, b)
ul_sb = ct.counting_upper_limit(nobs, b, CL, "CLsb")
ul_s = ct.counting_upper_limit(nobs, b, CL, "CLs")
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axs[0]
ax.plot(s, clsb, color=C[0], label="$CL_{s+b} = P(n \\leq n_{obs} | s+b)$")
ax.plot(s, np.full_like(s, clb), color=C[1], ls="--", label="$CL_b = P(n \\leq n_{obs} | b)$")
ax.plot(s, cls, color=C[2], ls="-.", label="$CL_s = CL_{s+b}/CL_b$")
ax.axhline(alpha, color="k", lw=0.8, ls=":")
for v, c in ((ul_sb, C[0]), (ul_s, C[2])):
    ax.axvline(v, color=c, lw=1, ls=":")
ax.set_yscale("log")
ax.set_ylim(1e-3, 1.5)
ax.set_xlabel("signal mean $s$")
ax.set_ylabel("confidence level")
ax.set_title(f"(a) b = {b:g}, $n_{{obs}}$ = {nobs}: "
             f"$s_{{up}}$ = {ul_sb:.2f} ($CL_{{s+b}}$), {ul_s:.2f} ($CL_s$)")
ax.legend(fontsize=8.5, loc="lower left")

print(f"== (a,b) counting experiment, b = {b}, {CL:.0%} CL ==")
print(f" n_obs = {nobs}: CL_b = {clb:.4f}; UL CL_s+b = {ul_sb:.3f}, UL CLs = {ul_s:.3f}")
print("  n_obs |  CL_s+b UL |   CLs UL  | Bayes flat-prior UL")
for n in range(0, 11):
    print(f"  {n:4d}  | {ct.counting_upper_limit(n, b, CL, 'CLsb'):9.3f}  | "
          f"{ct.counting_upper_limit(n, b, CL, 'CLs'):8.3f}  | {ct.counting_upper_limit(n, b, CL, 'bayes'):8.3f}")
print("  -> CLs = Bayes(flat prior on s) for every n and b; for n = 0 both give 2.996 whatever b is.")

# ---------------------------------------------------------------- (c) spurious exclusion
ax = axs[1]
nmax = int(b + 12 * np.sqrt(b) + 30)
ncount = np.arange(nmax)
pb = stats.poisson.pmf(ncount, b)
for bb, ls in ((b, "-"), (50.0, "--")):
    nmx = int(bb + 12 * np.sqrt(bb) + 30)
    nc = np.arange(nmx)
    w = stats.poisson.pmf(nc, bb)
    ul1 = ct.counting_upper_limit(nc, bb, CL, "CLsb")
    ul2 = ct.counting_upper_limit(nc, bb, CL, "CLs")
    ss = np.linspace(0.001, 4 * np.sqrt(bb) + 6, 300)
    p1 = [w[ul1 < x].sum() for x in ss]        # P(s excluded | background only)
    p2 = [w[ul2 < x].sum() for x in ss]
    ax.plot(ss / np.sqrt(bb), p1, color=C[0], ls=ls, label=f"$CL_{{s+b}}$, b = {bb:g}")
    ax.plot(ss / np.sqrt(bb), p2, color=C[2], ls=ls, label=f"$CL_s$, b = {bb:g}")
ax.axhline(alpha, color="k", lw=0.8, ls=":")
ax.set_xlabel("tested signal $s/\\sqrt{b}$")
ax.set_ylabel("P(exclude s | background only)")
ax.set_title("(c) exclusion power; small s: CL$_{s+b}$ $\\to$ 5%, CLs $\\to$ 0")
ax.legend(fontsize=8.5, loc="upper left")
ax.set_xlim(0, 4.5)
fig.savefig(f"{args.outdir}/03a_cls_counting.png")

ul_cls_n = ct.counting_upper_limit(ncount, b, CL, "CLs")
ul_sb_n = ct.counting_upper_limit(ncount, b, CL, "CLsb")
print(f"\n== (c) b = {b}: P(exclude s | b only) ==")
for x in (0.01, 0.5, 1, 2, 4, 6):
    print(f"  s = {x:5.2f}: CL_s+b {pb[ul_sb_n < x].sum():.4f}   CLs {pb[ul_cls_n < x].sum():.4f}")
print("  coverage of the CLs limit, P(s_true <= s_up):")
for x in (0.0, 1.0, 3.0, 6.0, 10.0):
    w = stats.poisson.pmf(ncount, x + b)
    print(f"  s_true = {x:4.1f}: CLs {w[ul_cls_n >= x].sum():.4f}   CL_s+b {w[ul_sb_n >= x].sum():.4f}  (nominal {CL})")

# ---------------------------------------------------------------- (d) expected limits
print(f"\n== (d) expected {CL:.0%} CLs limits: b-only toys ({args.ntoys}) vs asymptotic ==")
q = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
nsig = np.array([-2, -1, 0, 1, 2])
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
for ax, bb in zip(axs, (b, 100.0)):
    n_toy = rng.poisson(bb, args.ntoys)
    ul_toy = ct.counting_upper_limit(n_toy, bb, CL, "CLs")
    model = ct.BinnedModel([1.0], [bb], None)
    asym = np.array([ct.asymptotic_expected_limit(model, CL, k) for k in nsig])
    qt = np.quantile(ul_toy, q)
    print(f" b = {bb:g}")
    print("   quantile      " + "  ".join(f"{x:7.3f}" for x in q))
    print("   toys          " + "  ".join(f"{x:7.3f}" for x in qt))
    print("   asymptotic    " + "  ".join(f"{x:7.3f}" for x in asym))
    print(f"   mean of toys {ul_toy.mean():.3f}; with only integer n the toy limit is discrete")
    bins = np.linspace(0, np.quantile(ul_toy, 0.999) * 1.1, 80)
    ax.hist(ul_toy, bins, color=C[0], alpha=0.6, label="b-only toys: CLs UL")
    for v, k in zip(qt, nsig):
        ax.axvline(v, color=C[0], lw=1.6 if k == 0 else 1, ls="-" if k == 0 else "--")
    for v, k in zip(asym, nsig):
        ax.axvline(v, color=C[1], lw=1.6 if k == 0 else 1, ls="-" if k == 0 else ":")
    ax.plot([], [], color=C[0], label="toy median, $\\pm1\\sigma$, $\\pm2\\sigma$ quantiles")
    ax.plot([], [], color=C[1], label="asymptotic (Asimov) median and bands")
    ax.set_xlabel("95% CL upper limit on s")
    ax.set_ylabel("toys")
    ax.set_title(f"(d) distribution of upper limits, b = {bb:g}")
    ax.legend(fontsize=8, loc="upper right")
fig.savefig(f"{args.outdir}/03b_expected_limits_counting.png")

# ---------------------------------------------------------------- (e) Combine CLs by hand
print("\n== (e) Combine lhcClsByHand.cxx: N_obs = 4, B = 3.26, lnN kappa = 1.2353 on B ==")
model = ct.BinnedModel([1.0], [3.26], ("lnN", 1.2353))
n_obs, a_obs = np.array([[4.0]]), np.array([0.0])
mu_hat, th_hat, _ = model.fit(n_obs, a_obs)
print(f" best fit: r_hat = {mu_hat[0]:.3f}, theta_hat = {th_hat[0]:.3f}")
ntoy = max(args.ntoys // 2, 2000)
pt = ct.toy_cls_point(model, n_obs, a_obs, 1.0, ntoy, rng)
sig = ct.sigma_asimov(model, 1.0)
cls_as = ct.asymptotic_cls(pt["q_obs"], 1.0, sig)
print(f" r = 1: q~_obs = {pt['q_obs']:.4f}; toys ({ntoy}): p_mu = {pt['p_mu']:.4f}, "
      f"1-p_b = {pt['one_minus_pb']:.4f}, CLs = {pt['CLs']:.4f} +- {pt['CLs_err']:.4f}")
print(f"        asymptotic: p_mu = {cls_as[1]:.4f}, 1-p_b = {cls_as[2]:.4f}, CLs = {cls_as[0]:.4f}")
mu_grid = np.linspace(1.0, 12.0, 23)
scan = ct.toy_limit_scan(model, n_obs, a_obs, mu_grid, max(args.ntoys // 4, 2000), rng, CL)
print(f" 95% CL limit on r: toys {scan['obs']:.2f}, asymptotic "
      f"{ct.asymptotic_observed_limit(model, n_obs, a_obs, CL):.2f}")
print(f"   expected (toys, from grid)   : " +
      ", ".join(f"{k:.3f}: {v:.2f}" for k, v in scan["expected"].items()))
print(f"   expected (asymptotic)        : " +
      ", ".join(f"{k:+d}s: {ct.asymptotic_expected_limit(model, CL, k):.2f}" for k in nsig))
print(f"   no-systematic reference      : CLs UL (n=4, b=3.26) = {ct.counting_upper_limit(4, 3.26, CL, 'CLs'):.2f}")
# why is the limit with a systematic *lower* than the stat-only 6.31? Discreteness:
m_tiny = ct.BinnedModel([1.0], [3.26], ("lnN", 1.001))
pt6 = ct.toy_cls_point(m_tiny, n_obs, a_obs, 6.0, ntoy, rng)
print(f"   kappa -> 1.001, r = 6: toy CLs = {pt6['CLs']:.4f}  vs stat-only "
      f"P(n<=4|9.26)/P(n<=4|3.26) = {ct.counting_cls(4, 6.0, 3.26)[2]:.4f}\n"
      "   -> any continuous global observable breaks the ties between toys with n = n_obs,\n"
      "      so the toy p-values become ~P(n<n_obs) + P(n=n_obs)/2 ('mid-p') instead of\n"
      "      P(n<=n_obs): the stat-only discrete CLs limit is conservative.")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.errorbar(mu_grid, scan["CLs_obs"], [p["CLs_err"] for p in scan["points"]], fmt="o", ms=4,
            color=C[0], label="CLs from toys (q$\\tilde{}_\\mu$, frequentist toys)")
sig_grid = [ct.sigma_asimov(model, m) for m in mu_grid]
qobs = [model.test_statistics(m, n_obs, a_obs)["qtilde"][0] for m in mu_grid]
ax.plot(mu_grid, [ct.asymptotic_cls(qq, m, sg)[0] for qq, m, sg in zip(qobs, mu_grid, sig_grid)],
        color=C[1], ls="--", label="asymptotic CLs")
ax.axhline(alpha, color="k", lw=0.8, ls=":")
ax.set_yscale("log")
ax.set_xlabel("signal strength r (signal events)")
ax.set_ylabel("CLs")
ax.set_title("(e) N$_{obs}$ = 4, B = 3.26 $\\pm$ 23.5% (lnN)")
ax.legend(fontsize=8.5)
fig.savefig(f"{args.outdir}/03c_cls_by_hand.png")
print(f"\nplots written to {args.outdir}/03a_cls_counting.png, 03b_expected_limits_counting.png, 03c_cls_by_hand.png")
