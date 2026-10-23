#!/usr/bin/env python3
"""
05 - The other side of a limit: discovery significance with q_0, the Asimov
     median significance Z_A vs s/sqrt(b), and the look-elsewhere effect.

  (a) f(q_0 | 0) from background-only toys vs the asymptotic
      1/2 delta(q_0) + 1/2 chi2_1 (Wilks/Wald); Z = sqrt(q_0).
  (b) Median discovery significance of a counting experiment: toys vs
      Z_A = sqrt(2((s+b) ln(1+s/b) - s)) vs s/sqrt(b).
  (c) Look-elsewhere effect on the bump hunt of script 04: distribution of the
      maximum local significance over the mass range in background-only toys;
      global p-value of the observed excess.

Sources: book confidenceIntervals c17-c19, c22, c27; 2024/STAMET_May14_2024
p.13-33, p.46-47, p.52-58; papers/1007.1727.pdf Sec. 3.5, 4, 5.1;
C:part3/nonstandard L1116 (LEE with toys).

usage: python3 05_discovery_q0.py [--ntoys 20000] [--lee-toys 2000] [--outdir plots]
"""
import argparse
import os

import numpy as np
from scipy import stats

import cltoys as ct

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ntoys", type=int, default=20000)
ap.add_argument("--lee-toys", type=int, default=2000)
ap.add_argument("--seed", type=int, default=6)
ap.add_argument("--outdir", default="plots")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
rng = np.random.default_rng(args.seed)
plt = ct.setup_style()
C = ct.COLORS

# ------------------------------------------------------------------ (a) f(q0|0)
model, edges = ct.bump_model(125.0)
n, a = model.generate(0.0, 0.0, args.ntoys, rng)
q0, mu_hat = model.q0(n, a)
print("== (a) f(q0|0), bump model at 125 GeV, background-only toys ==")
print(f" fraction with q0 = 0 (mu_hat < 0): {np.mean(q0 == 0):.4f}  (asymptotic 0.5)")
for z in (1, 2, 3):
    print(f" P(q0 > {z * z}) = P(Z > {z}): toys {np.mean(q0 > z * z):.5f}, asymptotic {ct.z_to_p(z):.5f}")
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3))
ax = axs[0]
bins = np.linspace(0, 16, 65)
w = np.full(q0.size, 1 / q0.size / (bins[1] - bins[0]))
ax.hist(q0[q0 > 0], bins, weights=w[q0 > 0], color=C[0], alpha=0.5, label="toys, $q_0 > 0$")
qq = np.linspace(0.02, 16, 400)
ax.plot(qq, 0.5 * stats.chi2.pdf(qq, 1), color=C[1], label="$\\frac{1}{2}\\chi^2_1$ (Wilks)")
ax.set_yscale("log")
ax.set_xlabel("$q_0$")
ax.set_ylabel("density")
ax.set_title("(a) $f(q_0|0)$: $\\frac{1}{2}\\delta(q_0) + \\frac{1}{2}\\chi^2_1$;  $Z=\\sqrt{q_0}$")
ax.legend(fontsize=8.5)

# ------------------------------------------------------------------ (b) Z_A vs s/sqrt(b)
print("\n== (b) median discovery significance, counting experiment ==")
print("   b      s    s/sqrt(b)   Z_A    toys: median sqrt(q0)")
ax = axs[1]
for b, c in ((1.0, C[0]), (10.0, C[2]), (100.0, C[3])):
    ss = np.linspace(0.05, 6, 60) * np.sqrt(b) if b > 1 else np.linspace(0.05, 8, 60)
    za = np.sqrt(2 * ((ss + b) * np.log(1 + ss / b) - ss))
    ax.plot(ss / np.sqrt(b), za, color=c, label=f"$Z_A$, b = {b:g}")
    ax.plot(ss / np.sqrt(b), ss / np.sqrt(b), color=c, ls=":", lw=1.2)
    s_pts = np.array([1.0, 3.0, 5.0]) * np.sqrt(b)
    med = []
    for s in s_pts:
        nn = rng.poisson(s + b, args.ntoys)
        q = np.where(nn > b, 2 * (nn * np.log(np.maximum(nn, 1e-300) / b) + b - nn), 0.0)
        med.append(np.median(np.sqrt(q)))
        zA = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
        print(f" {b:5.0f} {s:6.2f}   {s / np.sqrt(b):6.2f}   {zA:6.3f}   {med[-1]:6.3f}")
    ax.plot(s_pts / np.sqrt(b), med, "o", color=c, mfc="white")
ax.plot([], [], "k:", lw=1.2, label="$s/\\sqrt{b}$")
ax.plot([], [], "ko", mfc="white", label="toys: median $\\sqrt{q_0}$")
ax.set_xlabel("$s/\\sqrt{b}$")
ax.set_ylabel("median significance")
ax.set_title("(b) $Z_A$ vs $s/\\sqrt{b}$ (s/$\\sqrt{b}$ overestimates at small b)")
ax.legend(fontsize=8)
fig.savefig(f"{args.outdir}/05a_discovery.png")

# ------------------------------------------------------------------ (c) LEE
masses = np.arange(105.0, 175.1, 2.5)
models = [ct.bump_model(m)[0] for m in masses]
data_n, data_a = ct.bump_model(150.0)[0].generate(1.5, 0.0, 1, np.random.default_rng(5))
z_obs = np.array([np.sqrt(mm.q0(data_n, data_a)[0][0]) for mm in models])
nb, ab = models[0].generate(0.0, 0.0, args.lee_toys, rng)     # background is the same for all masses
zmax = np.zeros(args.lee_toys)
for mm in models:
    zmax = np.maximum(zmax, np.sqrt(mm.q0(nb, ab)[0]))
z_loc = z_obs.max()
p_loc = ct.z_to_p(z_loc)
p_glob = np.mean(zmax >= z_loc)
print(f"\n== (c) look-elsewhere effect, {masses.size} mass points in [105, 175] GeV ==")
print(f" observed max local Z = {z_loc:.2f} at {masses[z_obs.argmax()]:g} GeV, local p = {p_loc:.2e}")
print(f" global p = P(max_m Z_loc >= {z_loc:.2f} | b only) = {p_glob:.4f} +- "
      f"{np.sqrt(p_glob * (1 - p_glob) / args.lee_toys):.4f} -> global Z = {ct.p_to_z(p_glob):.2f}; "
      f"trial factor ~ {p_glob / p_loc:.1f}")
fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.hist(zmax, np.linspace(0, 5, 51), color=C[0], alpha=0.6, label="b-only toys: max$_m$ local Z")
ax.axvline(z_loc, color="k", ls="--", label=f"observed max local Z = {z_loc:.2f}")
ax.set_xlabel("maximum local significance over the mass range")
ax.set_ylabel("toys")
ax.set_title(f"(c) LEE: p$_{{local}}$ = {p_loc:.1e}, p$_{{global}}$ = {p_glob:.3f}")
ax.legend(fontsize=8.5)
fig.savefig(f"{args.outdir}/05b_look_elsewhere.png")
print(f"\nplots written to {args.outdir}/05a_discovery.png, 05b_look_elsewhere.png")
