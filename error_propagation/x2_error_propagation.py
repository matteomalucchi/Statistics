#!/usr/bin/env python3
r"""
Toy study: propagation of a Gaussian uncertainty through y = x^2.

A measurement x ~ N(mu, sigma) is generated N_toys times. For each toy we compute
y = x^2, then compare the sample mean and standard deviation of y with

  1) linear (first-order) error propagation:
         <y>_lin   = mu^2
         sigma_lin = |dy/dx|_{x=mu} * sigma = 2 |mu| sigma
     Its p.d.f. comes from replacing y(x) by its tangent at x = mu,
         y_lin(x) = mu^2 + 2 mu (x - mu),
     an affine map of a Gaussian variable, so y_lin ~ N(mu^2, 2|mu|sigma) exactly
     (change of variables: f(y) = f_x(x(y)) |dx/dy| = phi((y-mu^2)/(2|mu|sigma)) / (2|mu|sigma)).
     The script also builds "linearised toys" by applying y_lin(x) to the same
     Gaussian toys x, which must reproduce that Gaussian.

  2) the exact moments of y (x Gaussian, so E[x^3] and E[x^4] are known):
         E[y]   = mu^2 + sigma^2
         Var[y] = E[x^4] - E[x^2]^2 = 4 mu^2 sigma^2 + 2 sigma^4

The exact p.d.f. of y (change of variables, two branches x = +-sqrt(y)) is
         f(y) = [ phi((sqrt(y)-mu)/sigma) + phi((-sqrt(y)-mu)/sigma) ] / (2 sigma sqrt(y)),  y > 0
i.e. sigma^2 times a non-central chi^2 with 1 d.o.f. and non-centrality (mu/sigma)^2.

The linear result is recovered when sigma/|mu| -> 0: the relative corrections are
  bias / sigma_lin   = sigma / (2|mu|)
  sigma_exact/sigma_lin = sqrt(1 + sigma^2 / (2 mu^2))

Usage:
    python3 x2_error_propagation.py --mu 10 --sigma 1 --ntoys 100000
    python3 x2_error_propagation.py --mu 1 --sigma 0.8          # non-linear regime
"""
import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# Analytic results
# --------------------------------------------------------------------------- #
def linear_propagation(mu, sigma):
    """First-order Taylor expansion of y = x^2 around x = mu."""
    return mu**2, 2.0 * abs(mu) * sigma


def linearised_y(x, mu):
    """Tangent of y = x^2 at x = mu: y_lin(x) = mu^2 + 2 mu (x - mu)."""
    return mu**2 + 2.0 * mu * (x - mu)


def exact_moments(mu, sigma):
    """Exact mean and standard deviation of y = x^2 for x ~ N(mu, sigma)."""
    mean = mu**2 + sigma**2
    var = 4.0 * mu**2 * sigma**2 + 2.0 * sigma**4
    return mean, np.sqrt(var)


def gauss_pdf(z):
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)


def exact_pdf_y(y, mu, sigma):
    """Exact p.d.f. of y = x^2 (defined for y > 0)."""
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    pos = y > 0
    s = np.sqrt(y[pos])
    out[pos] = (gauss_pdf((s - mu) / sigma) + gauss_pdf((-s - mu) / sigma)) / (2.0 * sigma * s)
    return out


# --------------------------------------------------------------------------- #
# Toy statistics with their own (statistical) uncertainties
# --------------------------------------------------------------------------- #
def sample_stats(y):
    """
    Sample mean and std with their standard errors.
      err(mean) = s / sqrt(N)
      err(s)    ~ sqrt( (m4 - s^4) / (4 s^2 N) )   (delta method, valid for any
                                                    distribution; y is NOT Gaussian,
                                                    so s/sqrt(2N) would be wrong)
    """
    n = y.size
    mean = y.mean()
    std = y.std(ddof=1)
    m4 = np.mean((y - mean) ** 4)
    err_mean = std / np.sqrt(n)
    err_std = np.sqrt((m4 - std**4) / (4.0 * std**2 * n))
    return mean, err_mean, std, err_std


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mu", type=float, default=5.0, help="true value of the measurement")
    p.add_argument("--sigma", type=float, default=1.0, help="Gaussian uncertainty on x")
    p.add_argument("--ntoys", type=int, default=100_000, help="number of toy measurements")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nbins", type=int, default=100)
    p.add_argument("--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "x2_error_propagation.png"))
    p.add_argument("--no-show", action="store_true", help="only save the figure")
    args = p.parse_args()

    mu, sigma, N = args.mu, args.sigma, args.ntoys
    rng = np.random.default_rng(args.seed)

    # ---- toys --------------------------------------------------------------
    x = rng.normal(mu, sigma, size=N)
    y = x**2                       # true transformation
    y_lin = linearised_y(x, mu)    # linearised transformation, same x toys

    m_toy, em_toy, s_toy, es_toy = sample_stats(y)
    m_tl, em_tl, s_tl, es_tl = sample_stats(y_lin)
    m_lin, s_lin = linear_propagation(mu, sigma)
    m_ex, s_ex = exact_moments(mu, sigma)

    # ---- printout ----------------------------------------------------------
    print(f"x ~ N(mu={mu}, sigma={sigma}),  sigma/|mu| = {sigma / abs(mu):.3f},  N_toys = {N}")
    print(f"{'':22s}{'mean(x^2)':>22s}{'sigma(x^2)':>22s}")
    print(f"{'toys, x^2':22s}{m_toy:>12.4f} +- {em_toy:<7.4f}{s_toy:>12.4f} +- {es_toy:<7.4f}")
    print(f"{'toys, linearised y':22s}{m_tl:>12.4f} +- {em_tl:<7.4f}{s_tl:>12.4f} +- {es_tl:<7.4f}")
    print(f"{'linear propagation':22s}{m_lin:>22.4f}{s_lin:>22.4f}")
    print(f"{'exact moments':22s}{m_ex:>22.4f}{s_ex:>22.4f}")
    print()
    print("Pulls  (toy - prediction) / stat. error of toy:")
    print(f"  vs linear : mean {(m_toy - m_lin) / em_toy:+8.2f}   sigma {(s_toy - s_lin) / es_toy:+8.2f}")
    print(f"  vs exact  : mean {(m_toy - m_ex) / em_toy:+8.2f}   sigma {(s_toy - s_ex) / es_toy:+8.2f}")
    print("  linearised toys vs linear propagation:"
          f" mean {(m_tl - m_lin) / em_tl:+8.2f}   sigma {(s_tl - s_lin) / es_tl:+8.2f}")

    # ---- figure ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (left) distribution of x^2
    lo = min(max(0.0, np.percentile(y, 0.05)), np.percentile(y_lin, 0.05))
    hi = max(np.percentile(y, 99.95), np.percentile(y_lin, 99.95))
    edges = np.linspace(lo, hi, args.nbins + 1)
    if lo < 0:  # put a bin edge exactly at 0 so the y = x^2 histogram is not smeared below 0
        edges = edges - edges[np.argmin(np.abs(edges))]
        edges = edges[(edges >= lo - 1e-12)]
    counts, edges, _ = ax1.hist(y, bins=edges, density=True,
                                histtype="stepfilled", alpha=0.35, color="C0",
                                label=f"toys: $y=x^2$ ($N={N}$)")
    ax1.hist(y_lin, bins=edges, density=True, histtype="step", lw=1.5, color="C3",
             label=r"toys: $y_{lin}=\mu^2+2\mu(x-\mu)$")
    # plotting range: also show y < 0, where the linearised Gaussian puts
    # probability that x^2 can never populate
    xmin = min(edges[0], m_lin - 3.5 * s_lin)
    yy = np.linspace(xmin, hi, 4000)
    yy = yy[np.abs(yy) > 1e-6]
    pdf_ex = exact_pdf_y(yy, mu, sigma)          # 0 for y < 0, ~1/sqrt(y) for y -> 0
    pdf_lin = gauss_pdf((yy - m_lin) / s_lin) / s_lin
    ax1.plot(yy, pdf_ex, "C0-", lw=2, label="exact p.d.f. of $x^2$")
    ax1.plot(yy, pdf_lin, "C3--", lw=2,
             label=r"linear prop.: $\mathcal{N}(\mu^2,\,2|\mu|\sigma)$")
    if m_lin - 3 * s_lin < 0:
        p_neg = 0.5 * (1.0 + math.erf(-m_lin / (s_lin * np.sqrt(2))))
        ax1.fill_between(yy[yy < 0], pdf_lin[yy < 0], color="C3", alpha=0.15,
                         label=f"lin. prop. P($y<0$) = {p_neg:.3f}")
    # the exact p.d.f. diverges at y = 0: cap the axis on the histogram / Gaussian
    # (skip the first bin, which contains the integrable 1/sqrt(y) spike)
    i0 = np.searchsorted(edges, 0.0, side="right")   # first bin with y > 0
    ymax = max(np.max(np.delete(counts, i0 - 1 if i0 > 0 else 0)), np.max(pdf_lin))
    ax1.set_ylim(0, 1.6 * ymax)
    ax1.set_xlim(xmin, hi)
    ax1.axvline(m_toy, color="k", lw=1.5, label=f"toy mean = {m_toy:.3f}")
    ax1.axvline(m_lin, color="C3", lw=1.5, ls=":", label=rf"$\mu^2$ = {m_lin:.3f}")
    ax1.axvline(m_ex, color="C0", lw=1.5, ls=":", label=rf"$\mu^2+\sigma^2$ = {m_ex:.3f}")
    ax1.set_xlabel("$y = x^2$")
    ax1.set_ylabel("probability density")
    ax1.set_title(rf"$x\sim\mathcal{{N}}({mu:g},\,{sigma:g})$,  $\sigma/|\mu|={sigma / abs(mu):.2f}$")
    txt = (f"{'':12s}{'mean':>9s}{'sigma':>9s}\n"
           f"{'toys x^2':12s}{m_toy:>9.3f}{s_toy:>9.3f}\n"
           f"{'exact':12s}{m_ex:>9.3f}{s_ex:>9.3f}\n"
           f"{'toys y_lin':12s}{m_tl:>9.3f}{s_tl:>9.3f}\n"
           f"{'linear':12s}{m_lin:>9.3f}{s_lin:>9.3f}")
    ax1.text(0.98, 0.55, txt, transform=ax1.transAxes, ha="right", va="top",
             family="monospace", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax1.legend(fontsize=8, loc="upper right")

    # (right) scan of sigma/|mu|: where does linear propagation break down?
    rel = np.linspace(0.02, 1.5, 30)
    n_scan = min(N, 50_000)
    r_mean, r_mean_err, r_std, r_std_err = [], [], [], []
    for r in rel:
        s_r = r * abs(mu)
        yt = rng.normal(mu, s_r, size=n_scan) ** 2
        mt, emt, st, est = sample_stats(yt)
        ml, sl = linear_propagation(mu, s_r)
        r_mean.append((mt - ml) / sl)          # bias in units of sigma_lin
        r_mean_err.append(emt / sl)
        r_std.append(st / sl)                  # width ratio
        r_std_err.append(est / sl)
    rr = np.linspace(rel[0], rel[-1], 300)
    ax2.errorbar(rel, r_std, yerr=r_std_err, fmt="o", ms=4, color="C0",
                 label=r"toys: $\sigma_{toy}/\sigma_{lin}$")
    ax2.plot(rr, np.sqrt(1 + rr**2 / 2), "C0-", label=r"exact: $\sqrt{1+\sigma^2/(2\mu^2)}$")
    ax2.errorbar(rel, r_mean, yerr=r_mean_err, fmt="s", ms=4, color="C1",
                 label=r"toys: $(\langle y\rangle_{toy}-\mu^2)/\sigma_{lin}$")
    ax2.plot(rr, rr / 2, "C1-", label=r"exact: $\sigma/(2|\mu|)$")
    ax2.axhline(1, color="C0", ls="--", lw=1)
    ax2.axhline(0, color="C1", ls="--", lw=1, label="linear propagation")
    ax2.axvline(sigma / abs(mu), color="0.5", ls=":", lw=1)
    ax2.set_xlabel(r"relative uncertainty $\sigma/|\mu|$")
    ax2.set_ylabel("ratio / bias w.r.t. linear propagation")
    ax2.set_title(f"Breakdown of linear propagation ({n_scan} toys per point)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nFigure saved to {args.output}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
