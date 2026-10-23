#!/usr/bin/env python3
"""
04 - LHC-style exclusion limits on a signal strength mu = sigma / sigma_ref
     with the profile likelihood ratio q~_mu and CLs: toys vs asymptotics,
     expected limits, +-1/2 sigma bands and the "Brazil plot" vs mass.

Prototype analysis (book confidenceIntervals c16): a binned m_X spectrum,
    E[n_i] = mu s_i(m_X) + b_i kappa^theta,   theta ~ N(a, 1)
exponential background (b_tot events in [100, 180] GeV, 40 bins), Gaussian
signal of width 2 GeV with s_tot events for mu = 1, and a log-normal
normalisation uncertainty on the background (kappa = 1.10). The background
shape is fixed; its normalisation is constrained by the sidebands and by the
auxiliary measurement a.

  (a) the "observed" pseudo-data (with a signal injected at --inject-mass) and
      the background-only fit.
  (b) at one mass: f(q~_mu | mu) and f(q~_mu | 0) from toys, with the
      asymptotic shapes, q~_obs, p_mu and 1 - p_b.
  (c) CLs(mu) from toys and from the asymptotic formula; observed limit,
      expected limit and bands ("expected from grid").
  (d) the Brazil plot: observed and expected 95% CL limits vs m_X, plus the
      local significance Z = sqrt(q_0) of the excess.

Sources: book confidenceIntervals c14-c20; 2024/STAMET_May14_2024 p.2-43;
papers/1007.1727.pdf Sec. 2-4; C:what_combine_does/statistical_tests L173;
C:part3/commonstatsmethods L7 (AsymptoticLimits), L297 (HybridNew);
Combine paper p.26-30.

usage: python3 04_profile_likelihood_limits.py [--ntoys 4000] [--mass 125]
         [--inject-mass 150] [--inject-mu 1.5] [--data-seed 5] [--toy-masses 120,150] [--outdir plots]
"""
import argparse
import os

import numpy as np

import cltoys as ct

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ntoys", type=int, default=4000, help="toys per tested mu (s+b and b-only each)")
ap.add_argument("--seed", type=int, default=4, help="seed for the toys")
ap.add_argument("--data-seed", type=int, default=5, help="seed for the pseudo-data")
ap.add_argument("--mass", type=float, default=125.0, help="mass hypothesis for panels (b), (c)")
ap.add_argument("--s-tot", type=float, default=20.0, help="signal events for mu = 1")
ap.add_argument("--b-tot", type=float, default=1000.0, help="background events in the window")
ap.add_argument("--kappa", type=float, default=1.10, help="lnN on the background normalisation")
ap.add_argument("--inject-mass", type=float, default=150.0)
ap.add_argument("--inject-mu", type=float, default=1.5)
ap.add_argument("--toy-masses", default="125,150", help="masses with toy limits on the Brazil plot ('' for none)")
ap.add_argument("--cl", type=float, default=0.95)
ap.add_argument("--outdir", default="plots")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
rng = np.random.default_rng(args.seed)
plt = ct.setup_style()
C = ct.COLORS
CL, alpha = args.cl, 1 - args.cl
mk = lambda m: ct.bump_model(m, s_tot=args.s_tot, b_tot=args.b_tot, nuisance=("lnN", args.kappa))

# ------------------------------------------------------------------ pseudo-data
m_inj, edges = mk(args.inject_mass)
n_obs, a_obs = m_inj.generate(args.inject_mu, 0.0, 1, np.random.default_rng(args.data_seed))
centres = 0.5 * (edges[1:] + edges[:-1])

model, _ = mk(args.mass)
fit = model.fit(n_obs, a_obs)
th_b, _ = model.profile_theta(0.0, n_obs, a_obs)
print(f"== pseudo-data: {int(n_obs.sum())} events, signal mu = {args.inject_mu} injected at "
      f"{args.inject_mass} GeV ==")
print(f" fit at m = {args.mass}: mu_hat = {fit[0][0]:.3f}, theta_hat = {fit[1][0]:.3f}")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.errorbar(centres, n_obs[0], np.sqrt(n_obs[0]), fmt="o", ms=3.5, color="k", label="pseudo-data")
ax.stairs(model.expected(np.array([0.0]), th_b)[0], edges, color=C[0], lw=2,
          label="background-only fit")
for m, c, ls in ((args.mass, C[1], "--"), (args.inject_mass, C[2], ":")):
    mm, _ = mk(m)
    ax.stairs(mm.expected(np.array([1.0]), th_b)[0], edges, color=c, ls=ls, lw=1.6,
              label=f"b + signal ($\\mu$ = 1) at {m:g} GeV")
ax.set_xlabel("$m_X$ [GeV]")
ax.set_ylabel("events / 2 GeV")
ax.set_title(f"(a) pseudo-data (signal injected at {args.inject_mass:g} GeV, $\\mu$ = {args.inject_mu})")
ax.legend(fontsize=8.5)
fig.savefig(f"{args.outdir}/04a_pseudodata.png")

# ------------------------------------------------------------------ (b) q~ distributions
mu_test = ct.asymptotic_expected_limit(model, CL, 0.0)       # near the expected limit
pt = ct.toy_cls_point(model, n_obs, a_obs, mu_test, args.ntoys, rng)
sig = ct.sigma_asimov(model, mu_test)
cls_a, pmu_a, pb_a = ct.asymptotic_cls(pt["q_obs"], mu_test, sig)
print(f"\n== (b) m = {args.mass}, tested mu = {mu_test:.3f} (sigma_A = {sig:.3f}) ==")
print(f" q~_obs = {pt['q_obs']:.3f}")
print(f" toys      : p_mu = {pt['p_mu']:.4f}, 1-p_b = {pt['one_minus_pb']:.4f}, CLs = {pt['CLs']:.4f} +- {pt['CLs_err']:.4f}")
print(f" asymptotic: p_mu = {pmu_a:.4f}, 1-p_b = {pb_a:.4f}, CLs = {cls_a:.4f}")

fig, ax = plt.subplots(figsize=(7.5, 4.4))
qmax = max(np.quantile(pt["q_b"], 0.995), np.quantile(pt["q_sb"], 0.995), pt["q_obs"]) * 1.15
bins = np.linspace(0, qmax, 60)
w = np.full(args.ntoys, 1.0 / args.ntoys / (bins[1] - bins[0]))
ax.hist(pt["q_sb"], bins, weights=w, color=C[0], alpha=0.45, label=f"toys $f(\\tilde q_\\mu|\\mu)$, s+b")
ax.hist(pt["q_b"], bins, weights=w, color=C[1], alpha=0.45, label="toys $f(\\tilde q_\\mu|0)$, b only")
qq = np.linspace(1e-4, qmax, 800)
for mp, c, lab in ((mu_test, C[0], "asymptotic, s+b"), (0.0, C[1], "asymptotic, b only")):
    F = ct.qtilde_cdf_asym(qq, mu_test, mp, sig)
    ax.plot(qq[1:], np.diff(F) / np.diff(qq), color=c, lw=1.6, label=lab)
ax.axvline(pt["q_obs"], color="k", ls="--", lw=1.4, label="$\\tilde q_{\\mu,obs}$")
ax.set_yscale("log")
ax.set_ylim(1e-4, 5)
ax.set_xlabel("$\\tilde q_\\mu$")
ax.set_ylabel("probability density")
ax.set_title(f"(b) m = {args.mass:g} GeV, $\\mu$ = {mu_test:.2f}: "
             f"CLs = {pt['CLs']:.3f} (toys), {cls_a:.3f} (asympt.)")
ax.legend(fontsize=8.5)
fig.savefig(f"{args.outdir}/04b_qtilde_distributions.png")

# ------------------------------------------------------------------ (c) CLs scan
mu_grid = np.linspace(0.2, 3.0, 15) * ct.asymptotic_expected_limit(model, CL, 0.0)
scan = ct.toy_limit_scan(model, n_obs, a_obs, mu_grid, args.ntoys, rng, CL)
sig_g = np.array([ct.sigma_asimov(model, m) for m in mu_grid])
q_g = np.array([p["q_obs"] for p in scan["points"]])
cls_asym = ct.asymptotic_cls(q_g, mu_grid, sig_g)[0]
obs_a = ct.asymptotic_observed_limit(model, n_obs, a_obs, CL)
exp_a = {k: ct.asymptotic_expected_limit(model, CL, k) for k in (-2, -1, 0, 1, 2)}
print(f"\n== (c) {CL:.0%} CL limits on mu at m = {args.mass} ==")
print(f" observed  : toys {scan['obs']:.3f}, asymptotic {obs_a:.3f}")
print("  expected  quantile:   2.5%    16%     50%     84%    97.5%")
print("            toys     : " + "  ".join(f"{v:6.3f}" for v in scan["expected"].values()))
print("            asympt.  : " + "  ".join(f"{v:6.3f}" for v in exp_a.values()))

fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.errorbar(mu_grid, scan["CLs_obs"], [p["CLs_err"] for p in scan["points"]], fmt="o", ms=4,
            color=C[0], label="observed CLs, toys")
ax.plot(mu_grid, cls_asym, color=C[0], ls="--", label="observed CLs, asymptotic")
for k, ls in ((0.5, "-"), (0.16, ":"), (0.84, ":")):
    vals = []
    for p in scan["points"]:
        qk = np.quantile(p["q_b"], 1 - k)
        vals.append(np.mean(p["q_sb"] >= qk) / max(np.mean(p["q_b"] >= qk), 1e-12))
    ax.plot(mu_grid, vals, color=C[1], ls=ls, lw=1.4,
            label="expected CLs (median, toys)" if k == 0.5 else ("$\\pm1\\sigma$ (toys)" if k == 0.16 else None))
ax.axhline(alpha, color="k", lw=0.8, ls=":")
ax.axvline(scan["obs"], color=C[0], lw=1, ls=":")
ax.set_yscale("log")
ax.set_ylim(1e-3, 1.3)
ax.set_xlabel("signal strength $\\mu$")
ax.set_ylabel("CLs")
ax.set_title(f"(c) CLs scan at m = {args.mass:g} GeV")
ax.legend(fontsize=8.5, loc="lower left")
fig.savefig(f"{args.outdir}/04c_cls_scan.png")

# ------------------------------------------------------------------ (d) Brazil plot
masses = np.arange(105.0, 175.1, 2.5)
obs, med, m1, p1, m2, p2, z0, muhat = ([] for _ in range(8))
for m in masses:
    mm, _ = mk(m)
    obs.append(ct.asymptotic_observed_limit(mm, n_obs, a_obs, CL))
    e = [ct.asymptotic_expected_limit(mm, CL, k) for k in (-2, -1, 0, 1, 2)]
    m2.append(e[0]); m1.append(e[1]); med.append(e[2]); p1.append(e[3]); p2.append(e[4])
    q0, mh = mm.q0(n_obs, a_obs)
    z0.append(np.sqrt(q0[0])); muhat.append(mh[0])
toy_m = [float(x) for x in args.toy_masses.split(",") if x.strip()]
toy_obs = []
for m in toy_m:
    mm, _ = mk(m)
    grid = np.linspace(0.2, 3.5, 12) * ct.asymptotic_expected_limit(mm, CL, 0.0)
    sc = ct.toy_limit_scan(mm, n_obs, a_obs, grid, args.ntoys // 2, rng, CL)
    toy_obs.append(sc["obs"])

fig, (ax, axz) = plt.subplots(2, 1, figsize=(7.5, 6.6), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1]})
ax.fill_between(masses, m2, p2, color=ct.BRAZIL_2S, label="expected $\\pm2\\sigma$")
ax.fill_between(masses, m1, p1, color=ct.BRAZIL_1S, label="expected $\\pm1\\sigma$")
ax.plot(masses, med, color="k", ls="--", lw=1.4, label="expected (median, Asimov)")
ax.plot(masses, obs, color="k", marker="o", ms=3, lw=1.6, label="observed (asymptotic)")
if toy_m:
    ax.plot(toy_m, toy_obs, "D", color=C[1], ms=7, mfc="white", mew=2, label="observed (toys)")
ax.axhline(1.0, color=C[0], lw=1.2, label="$\\mu$ = 1 (reference signal)")
ax.set_ylabel(f"{CL:.0%} CL upper limit on $\\mu$")
ax.set_title("(d) exclusion limits vs mass (CLs, $\\tilde q_\\mu$)")
ax.legend(fontsize=8, ncol=2, loc="upper left")
ax.set_ylim(0, max(p2) * 1.6)
axz.plot(masses, z0, color=C[2], marker="o", ms=3)
axz.set_ylabel("local $Z = \\sqrt{q_0}$")
axz.set_xlabel("$m_X$ hypothesis [GeV]")
for z in (1, 2, 3):
    axz.axhline(z, color="0.6", lw=0.7, ls=":")
fig.savefig(f"{args.outdir}/04d_brazil_plot.png")

print(f"\n== (d) Brazil plot (asymptotic) ==")
print("  m [GeV]  observed  expected [-2s, -1s, med, +1s, +2s]          mu_hat   local Z")
for i, m in enumerate(masses):
    if i % 2 == 0:
        print(f"  {m:6.1f}   {obs[i]:6.3f}    [{m2[i]:.3f}, {m1[i]:.3f}, {med[i]:.3f}, {p1[i]:.3f}, {p2[i]:.3f}]   "
              f"{muhat[i]:+.3f}   {z0[i]:.2f}")
for m, t in zip(toy_m, toy_obs):
    i = int(np.argmin(np.abs(masses - m)))
    print(f"  toys at {m:g} GeV: observed limit {t:.3f} (asymptotic {obs[i]:.3f})")
print(f"\nplots written to {args.outdir}/04a_pseudodata.png ... 04d_brazil_plot.png")
