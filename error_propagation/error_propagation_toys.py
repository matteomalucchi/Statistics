#!/usr/bin/env python3
r"""
Toy study of error propagation through a function y = f(x).

A measurement x is drawn from an input distribution with mean mu and standard
deviation sigma:
    gauss   : x ~ N(mu, sigma)
    uniform : x ~ U(mu - sqrt(3) sigma, mu + sqrt(3) sigma)       (std = sigma)
    poisson : x ~ Poisson(lambda = mu)        (sigma = sqrt(mu); --sigma ignored)

For every toy x_i we compute
    y_i     = f(x_i)                                  (true transformation)
    y_lin,i = f(mu) + f'(mu) (x_i - mu)               (tangent at x = mu)
and compare their sample mean / std with

  * linear error propagation:   <y> = f(mu),  sigma_y = |f'(mu)| sigma
    If one also assumes a Gaussian shape, its p.d.f. is N(f(mu), |f'(mu)| sigma).
    This shape is the exact p.d.f. of y_lin only when x itself is Gaussian (an
    affine map of a Gaussian is Gaussian). For uniform or Poisson x, y_lin is
    uniform or discrete: the propagation formula fixes only the first two moments.

  * the "exact" result: E[f(x)] and Var[f(x)] computed by numerical integration
    over the input p.d.f. (a fine midpoint grid for continuous x, a direct sum over
    k for Poisson). The exact distribution of y is shown bin by bin by sending the
    same quadrature through f with the same binning as the toys.

The "exact" histogram (solid blue line) is the bin-averaged p.d.f. of y,
    h_b = P(y in bin b) / Delta y_b = (1/Delta y_b) * integral over {x : f(x) in b} of f_X(x) dx ,
obtained by pushing the quadrature nodes x_j, with weights w_j = f_X(x_j) Delta x
(Poisson: w_k = P(k)), through f and histogramming the f(x_j) with weights w_j in
the same bins as the toys. This needs neither the inverse of f nor its
derivative: it is the change-of-variables formula
f_Y(y) = sum over roots x_r of f(x_r) = y of f_X(x_r)/|f'(x_r)| integrated over each bin.
The quadrature error is one grid cell (weight f_X * Delta x, Delta x = 1e-4 sigma)
misassigned per bin edge: ~1e-3 relative per bin for 100 bins, well below the toy
fluctuations.

Whenever possible (built-in functions, continuous input) the plotted exact p.d.f. is
instead ANALYTIC, by change of variables over all inverse branches x_r(y):
    f_Y(y) = sum_r f_X(x_r(y)) |dx_r/dy|   / P(x in domain of f)
e.g. x^2: [f_X(sqrt y) + f_X(-sqrt y)]/(2 sqrt y); e^x: f_X(ln y)/y (log-normal);
sin x: sum over n of both arcsin branches shifted by 2 pi n, / sqrt(1 - y^2).
It is cross-checked against the numerical bin averages (printed). For Poisson x the
exact distribution P(y = f(k)) = Poisson(k; lambda) is analytic already. Only free
expressions (--funcs "...") fall back to the numerical bin-averaged version.

Coverage of the +-1 sigma interval: the probability that the true y = f(x) lies in
    [E[y] - sigma_y, E[y] + sigma_y]                   (exact moments)
    [f(mu) - |f'(mu)| sigma, f(mu) + |f'(mu)| sigma]   (linear propagation)
computed both from the exact distribution (quadrature) and from the toys, and
compared with the Gaussian value 68.27%. The linear interval applied to y_lin is
also shown. It equals P(|x - mu| <= sigma) of the input (68.27% Gaussian,
1/sqrt(3) = 57.7% uniform, lambda-dependent for Poisson), so even a perfectly
linear f gives 68% only if x is Gaussian.

Per-toy (frequentist) coverage: each toy is one experiment that measures x_i
and quotes, by linear propagation evaluated at the MEASURED value,
    y_i = f(x_i) +- |f'(x_i)| sigma .
The coverage is the fraction of experiments whose interval contains the true
value f(mu). (sigma is the known resolution of x; for Poisson the true sqrt(mu)
is used, not the sqrt(k_i) an experimenter would plug in.) For linear f this
is again P(|x - mu| <= sigma). Intervals are closed, which matters
for discrete (Poisson) x, where y can sit exactly on an edge.

Domain: toys or quadrature nodes where f(x) is not finite (log of x <= 0, sqrt of
x < 0, 1/0, ...) are dropped and the rest is renormalised, i.e. the input is
truncated to the domain of f. The dropped fraction is printed.

Non-existent moments: if f has a pole inside the support of a continuous input
(1/x with Gaussian x: f_X(0) > 0, so the integral of f_X(x)/x dx diverges), then
E[f(x)] and Var[f(x)] do not exist. The script reports this rather than a
grid-dependent number. The p.d.f. of y still exists and is shown. The toy
mean/std are then finite only because a finite sample never lands close enough
to the pole; they do not converge as N grows. For a uniform input the support is
bounded, so 1/x has finite moments as long as 0 lies outside [a, b]. Poles are
declared only for the built-in functions, not for expressions.

Functions: built-in names (see --list-funcs) or any numpy expression in x, e.g.
    --funcs x2 log "np.arctan(x)" "x**3 - 2*x"
(For expressions the derivative is taken numerically. They are evaluated with
eval(), so only pass trusted input.)

Examples:
    python3 error_propagation_toys.py                         # default set, Gaussian
    python3 error_propagation_toys.py --dist uniform
    python3 error_propagation_toys.py --dist poisson --mu 9
    python3 error_propagation_toys.py --funcs x2 --mu 1 --sigma 0.8
"""
import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np


# =========================================================================== #
# Functions
# =========================================================================== #
@dataclass
class Func:
    name: str
    f: Callable
    df: Optional[Callable]      # analytic derivative (None -> numerical)
    label: str                  # label for plots (mathtext allowed)
    scan_max: float = 1.0       # max sigma/|mu| in the breakdown scan
    note: str = ""
    poles: tuple = ()           # x where f diverges so that E[f] or E[f^2] is infinite
    branches: Optional[Callable] = None   # y, xlo, xhi -> [(x_r(y), |dx_r/dy|), ...]
    domain: tuple = (-math.inf, math.inf)  # x-range where f is defined (for normalisation)
    pdf_formula: str = ""

    def deriv_vec(self, x):
        """f'(x) for an array of x (analytic if available, else central difference)."""
        x = np.asarray(x, dtype=float)
        if self.df is not None:
            return np.asarray(self.df(x), dtype=float) * np.ones_like(x)
        h = 1e-5 * np.maximum(1.0, np.abs(x))
        return (self.f(x + h) - self.f(x - h)) / (2 * h)

    def deriv(self, x0):
        if self.df is not None:
            return float(self.df(x0))
        h = 1e-5 * max(1.0, abs(x0))
        return float((self.f(x0 + h) - self.f(x0 - h)) / (2 * h))


# Inverse branches for the analytic p.d.f.  f_Y(y) = sum_r f_X(x_r(y)) |dx_r/dy|.
# Each returns a list of (x_r(y), |dx_r/dy|) arrays; NaN where y is outside the range.
def _where(cond, v):
    return np.where(cond, v, np.nan)


def _br_x2(y, *_):
    s = np.sqrt(_where(y > 0, y))
    return [(s, 0.5 / s), (-s, 0.5 / s)]


def _br_inv(y, *_):
    yy = _where(y != 0, y)
    return [(1.0 / yy, 1.0 / yy**2)]


def _br_sqrt(y, *_):
    yy = _where(y >= 0, y)
    return [(yy**2, 2.0 * yy)]


def _br_log(y, *_):
    return [(np.exp(y), np.exp(y))]


def _br_exp(y, *_):
    yy = _where(y > 0, y)
    return [(np.log(yy), 1.0 / yy)]


def _br_x3(y, *_):
    c = np.cbrt(y)
    return [(c, _where(y != 0, 1.0 / (3.0 * c**2)))]


def _br_sin(y, xlo, xhi):
    """x = arcsin(y) + 2 pi n  and  x = pi - arcsin(y) + 2 pi n, for all n reaching [xlo, xhi]."""
    yy = _where(np.abs(y) < 1, y)
    a, jac = np.arcsin(yy), 1.0 / np.sqrt(1.0 - yy**2)
    out = []
    for n in range(math.floor((xlo - math.pi) / (2 * math.pi)) - 1, math.ceil(xhi / (2 * math.pi)) + 2):
        out += [(a + 2 * math.pi * n, jac), (math.pi - a + 2 * math.pi * n, jac)]
    return out


FUNCS = {
    "x2":   Func("x2",   lambda x: x**2,      lambda x: 2 * x,               r"$x^2$",       1.5,
                 "bias of the mean is exactly +sigma^2",
                 branches=_br_x2, pdf_formula="[f_X(sqrt y) + f_X(-sqrt y)] / (2 sqrt y),  y > 0"),
    "inv":  Func("inv",  lambda x: 1.0 / x,   lambda x: -1.0 / x**2,          r"$1/x$",       0.55,
                 "e.g. momentum from curvature; E[1/x] diverges if f_X(0) > 0", poles=(0.0,),
                 branches=_br_inv, pdf_formula="f_X(1/y) / y^2"),
    "sqrt": Func("sqrt", np.sqrt,             lambda x: 0.5 / np.sqrt(x),     r"$\sqrt{x}$",  0.6,
                 "e.g. amplitude from a counted rate",
                 branches=_br_sqrt, domain=(0.0, math.inf), pdf_formula="2 y f_X(y^2) / P(x >= 0),  y >= 0"),
    "log":  Func("log",  np.log,              lambda x: 1.0 / x,              r"$\ln x$",     0.5,
                 "bias ~ -sigma^2/(2 mu^2) at second order",
                 branches=_br_log, domain=(0.0, math.inf), pdf_formula="e^y f_X(e^y) / P(x > 0)"),
    "exp":  Func("exp",  np.exp,              np.exp,                         r"$e^{x}$",     1.0,
                 "Gaussian x -> log-normal y",
                 branches=_br_exp, pdf_formula="f_X(ln y) / y,  y > 0  (log-normal for Gaussian x)"),
    "sin":  Func("sin",  np.sin,              np.cos,                         r"$\sin x$",    1.5,
                 "periodic, non-monotonic",
                 branches=_br_sin,
                 pdf_formula="sum_n [f_X(asin y + 2 pi n) + f_X(pi - asin y + 2 pi n)] / sqrt(1 - y^2)"),
    "x3":   Func("x3",   lambda x: x**3,      lambda x: 3 * x**2,             r"$x^3$",       1.5,
                 "odd power",
                 branches=_br_x3, pdf_formula="f_X(cbrt y) / (3 cbrt(y)^2)"),
}
DEFAULT_FUNCS = ["x2", "inv", "sqrt", "log", "exp"]


def make_func(spec):
    if spec in FUNCS:
        return FUNCS[spec]
    ns = {k: getattr(np, k) for k in ("sin", "cos", "tan", "exp", "log", "log10", "sqrt",
                                      "arctan", "arcsin", "arccos", "sinh", "cosh", "tanh", "abs", "pi")}
    ns["np"] = np

    def f(x, _e=spec):
        return eval(_e, {"__builtins__": {}}, {**ns, "x": x})

    f(np.array([1.0, 2.0]))  # fail early on a bad expression
    return Func(re.sub(r"[^A-Za-z0-9]+", "_", spec).strip("_"), f, None, f"f(x) = {spec}", 1.0)


# =========================================================================== #
# Input distributions
# =========================================================================== #
class InputDist:
    """Toy generator + quadrature (nodes, weights) of the input p.d.f."""

    def __init__(self, kind, mu, sigma):
        self.kind, self.mu = kind, float(mu)
        if kind == "poisson":
            if mu <= 0:
                raise ValueError("Poisson mean must be > 0")
            self.sigma = math.sqrt(mu)
        else:
            self.sigma = float(sigma)
        self.discrete = kind == "poisson"

    def support(self):
        if self.kind == "gauss":
            return self.mu - 10 * self.sigma, self.mu + 10 * self.sigma
        h = math.sqrt(3) * self.sigma
        return self.mu - h, self.mu + h

    def pdf(self, x):
        """Analytic p.d.f. of a continuous input."""
        x = np.asarray(x, dtype=float)
        if self.kind == "gauss":
            return np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (math.sqrt(2 * math.pi) * self.sigma)
        h = math.sqrt(3) * self.sigma
        return np.where(np.abs(x - self.mu) <= h, 1.0 / (2 * h), 0.0)

    def cdf(self, x):
        if math.isinf(x):
            return 0.0 if x < 0 else 1.0
        if self.kind == "gauss":
            return 0.5 * (1 + math.erf((x - self.mu) / (math.sqrt(2) * self.sigma)))
        h = math.sqrt(3) * self.sigma
        return min(1.0, max(0.0, (x - self.mu + h) / (2 * h)))

    @property
    def label(self):
        if self.kind == "gauss":
            return rf"$x\sim\mathcal{{N}}({self.mu:g},\,{self.sigma:g})$"
        if self.kind == "uniform":
            h = math.sqrt(3) * self.sigma
            return rf"$x\sim\mathcal{{U}}[{self.mu - h:.3g},\,{self.mu + h:.3g}]$"
        return rf"$x\sim\mathrm{{Poisson}}({self.mu:g})$"

    def sample(self, rng, n):
        if self.kind == "gauss":
            return rng.normal(self.mu, self.sigma, n)
        if self.kind == "uniform":
            h = math.sqrt(3) * self.sigma
            return rng.uniform(self.mu - h, self.mu + h, n)
        return rng.poisson(self.mu, n).astype(float)

    def quadrature(self, n=200_001):
        """Nodes and normalised weights such that E[g(x)] ~= sum_i w_i g(x_i)."""
        if self.kind == "gauss":
            edges = np.linspace(self.mu - 10 * self.sigma, self.mu + 10 * self.sigma, n + 1)
            x = 0.5 * (edges[1:] + edges[:-1])
            w = np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)
        elif self.kind == "uniform":
            h = math.sqrt(3) * self.sigma
            edges = np.linspace(self.mu - h, self.mu + h, n + 1)
            x = 0.5 * (edges[1:] + edges[:-1])
            w = np.ones_like(x)
        else:
            kmax = int(self.mu + 15 * self.sigma + 30)
            x = np.arange(kmax + 1, dtype=float)
            lg = np.array([math.lgamma(k + 1) for k in range(kmax + 1)])
            w = np.exp(x * math.log(self.mu) - self.mu - lg)
        return x, w / w.sum()


# =========================================================================== #
# Statistics
# =========================================================================== #
def pole_in_support(fn, dist):
    """True if a (declared) pole of f lies inside the support of a continuous input."""
    if dist.discrete or not fn.poles:
        return False
    if dist.kind == "gauss":
        return True                                   # unbounded support
    h = math.sqrt(3) * dist.sigma
    return any(dist.mu - h < p0 < dist.mu + h for p0 in fn.poles)


def analytic_pdf_y(fn, dist, y):
    """
    Exact p.d.f. of y = f(x) by change of variables, f_Y(y) = sum_r f_X(x_r) |dx_r/dy|,
    renormalised to the domain of f (as the toys are). None if not available.
    """
    if dist.discrete or fn.branches is None:
        return None
    xlo, xhi = dist.support()
    out = np.zeros_like(np.asarray(y, dtype=float))
    for xr, jac in fn.branches(np.asarray(y, dtype=float), xlo, xhi):
        out += np.nan_to_num(dist.pdf(xr) * jac, nan=0.0, posinf=0.0)
    p_dom = dist.cdf(fn.domain[1]) - dist.cdf(fn.domain[0])
    return out / p_dom


def finite_eval(f, x):
    with np.errstate(all="ignore"):
        y = np.asarray(f(x), dtype=float)
    return y, np.isfinite(y)


def exact_moments(f, x, w):
    """Mean, std of f(x) under the (domain-truncated) input p.d.f. + lost probability."""
    y, ok = finite_eval(f, x)
    lost = w[~ok].sum()
    wv = w[ok] / w[ok].sum()
    m = np.sum(wv * y[ok])
    v = np.sum(wv * (y[ok] - m) ** 2)
    return m, math.sqrt(v), lost


GAUSS_1SIGMA = math.erf(1 / math.sqrt(2))     # 0.682689...


def in_interval(y, lo, hi):
    """Closed interval with a relative tolerance (discrete y can sit on the edge)."""
    tol = 1e-12 * max(1.0, abs(lo), abs(hi))
    return (y >= lo - tol) & (y <= hi + tol)


def coverage_exact(y, w, lo, hi):
    """P(lo <= y <= hi) for the quadrature (y_i, w_i), w normalised."""
    return float(np.sum(w[in_interval(y, lo, hi)]))


def coverage_toys(y, lo, hi):
    """Fraction of toys in [lo, hi] and its binomial error."""
    p = float(np.mean(in_interval(y, lo, hi)))
    return p, math.sqrt(p * (1 - p) / y.size)


def per_toy_cover(fn, x, sigma, y_true):
    """Boolean per x: does f(x) +- |f'(x)| sigma contain y_true? (non-finite -> dropped)"""
    y, ok = finite_eval(fn.f, x)
    with np.errstate(all="ignore"):
        half = np.abs(fn.deriv_vec(x)) * sigma
    ok &= np.isfinite(half)
    tol = 1e-12 * max(1.0, abs(y_true))
    return (np.abs(y - y_true) <= half + tol), ok


def sample_stats(y):
    """
    Sample mean and std with standard errors:
      err(mean) = s / sqrt(N)
      err(s)   ~= sqrt((m4 - s^4) / (4 s^2 N))   (delta method; valid for any
                                                 distribution, y is in general not Gaussian)
    """
    n = y.size
    mean = y.mean()
    s = y.std(ddof=1)
    m4 = np.mean((y - mean) ** 4)
    return mean, s / math.sqrt(n), s, math.sqrt(max(m4 - s**4, 0.0) / (4 * s**2 * n))


def weighted_quantile(v, w, q):
    i = np.argsort(v)
    c = np.cumsum(w[i])
    return np.interp(np.asarray(q) * c[-1], c, v[i])


def gauss_pdf(z):
    return np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)


def lattice_edges(values):
    """Bin edges centred on each distinct value (for discrete y)."""
    v = np.unique(values)
    if v.size == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    mid = 0.5 * (v[1:] + v[:-1])
    return np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])


# =========================================================================== #
# One function: toys, table, figure
# =========================================================================== #
def run_one(fn, dist, args, rng):
    mu, sigma, N = dist.mu, dist.sigma, args.ntoys

    # ---- toys -------------------------------------------------------------
    x = dist.sample(rng, N)
    y, ok = finite_eval(fn.f, x)
    lost_toys = 1.0 - ok.mean()
    y = y[ok]
    f0, d0 = float(fn.f(mu)), fn.deriv(mu)
    y_lin = f0 + d0 * (x - mu)              # linearised: never out of domain

    m_t, em_t, s_t, es_t = sample_stats(y)
    m_tl, em_tl, s_tl, es_tl = sample_stats(y_lin)
    m_lin, s_lin = f0, abs(d0) * sigma
    xq, wq = dist.quadrature()
    m_ex, s_ex, lost_ex = exact_moments(fn.f, xq, wq)
    has_exact = not pole_in_support(fn, dist)
    if not has_exact:
        m_ex = s_ex = float("nan")

    # ---- +-1 sigma coverage ---------------------------------------------------
    yq_all, okq_all = finite_eval(fn.f, xq)
    yq_v, wq_v = yq_all[okq_all], wq[okq_all] / wq[okq_all].sum()
    ylq_all = f0 + d0 * (xq - mu)
    lin_lo, lin_hi = m_lin - s_lin, m_lin + s_lin
    cov = {}   # name -> (exact, toys, toys_err)
    if has_exact:
        cov["ex"] = (coverage_exact(yq_v, wq_v, m_ex - s_ex, m_ex + s_ex),
                     *coverage_toys(y, m_ex - s_ex, m_ex + s_ex))
    cov["lin"] = (coverage_exact(yq_v, wq_v, lin_lo, lin_hi), *coverage_toys(y, lin_lo, lin_hi))
    cov["ylin"] = (coverage_exact(ylq_all, wq, lin_lo, lin_hi), *coverage_toys(y_lin, lin_lo, lin_hi))
    # per-toy interval f(x_i) +- |f'(x_i)| sigma containing the true f(mu)
    cq, okc = per_toy_cover(fn, xq, sigma, f0)
    ct, okt = per_toy_cover(fn, x, sigma, f0)
    pt = float(ct[okt].mean())
    cov["toy"] = (float(np.sum(wq[okc & cq]) / np.sum(wq[okc])), pt, math.sqrt(pt * (1 - pt) / okt.sum()))

    # ---- printout ---------------------------------------------------------
    print("=" * 78)
    print(f"f(x) = {fn.name}   [{fn.note}]" if fn.note else f"f(x) = {fn.name}")
    print(f"input: {dist.kind}, mu = {mu:g}, sigma = {sigma:.4g}, sigma/|mu| = {sigma / abs(mu):.3f}, N = {N}")
    if lost_toys > 0 or lost_ex > 1e-12:
        print(f"  WARNING: x outside the domain of f, dropped: toys {lost_toys:.2e}, exact {lost_ex:.2e}")
    print(f"{'':24s}{'mean(y)':>22s}{'sigma(y)':>22s}")
    print(f"{'toys, y = f(x)':24s}{m_t:>12.5g} +- {em_t:<7.2g}{s_t:>12.5g} +- {es_t:<7.2g}")
    if has_exact:
        print(f"{'exact (numerical)':24s}{m_ex:>22.5g}{s_ex:>22.5g}")
    else:
        print(f"{'exact':24s}{'does not exist':>22s}{'does not exist':>22s}"
              "   (pole of f inside the support of x)")
    print(f"{'toys, y_lin':24s}{m_tl:>12.5g} +- {em_tl:<7.2g}{s_tl:>12.5g} +- {es_tl:<7.2g}")
    print(f"{'linear propagation':24s}{m_lin:>22.5g}{s_lin:>22.5g}")
    print(f"+-1 sigma coverage (Gaussian: {100 * GAUSS_1SIGMA:.2f}%)"
          f"{'exact':>16s}{'toys':>18s}")
    rows = [("ex", "[E[y] +- sigma_y]      on y = f(x)"),
            ("lin", "[f(mu) +- sigma_lin]   on y = f(x)"),
            ("ylin", "[f(mu) +- sigma_lin]   on y_lin"),
            ("toy", "f(x_i) +- |f'(x_i)|sigma  covers f(mu)")]
    for k, lab in rows:
        if k in cov:
            c, cty, ect = cov[k]
            print(f"  {lab:36s}{100 * c:>8.2f}%{100 * cty:>11.2f} +- {100 * ect:.2f}%")
        else:
            print(f"  {lab:36s}{'n/a (moments do not exist)':>30s}")
    print("pulls (toy - prediction)/err:")
    print(f"  f(x) toys vs linear : mean {(m_t - m_lin) / em_t:+9.2f}   sigma {(s_t - s_lin) / es_t:+9.2f}")
    if has_exact:
        print(f"  f(x) toys vs exact  : mean {(m_t - m_ex) / em_t:+9.2f}   sigma {(s_t - s_ex) / es_t:+9.2f}")
    if has_exact and abs(s_t - s_ex) / es_t > 10:
        print("  NOTE: sample std is far from the exact one -> heavy tail: the sample moments\n"
              "        have not converged at this N (the error estimate itself is unreliable)")
    if s_lin > 0:
        print(f"  y_lin toys vs linear: mean {(m_tl - m_lin) / em_tl:+9.2f}   sigma {(s_tl - s_lin) / es_tl:+9.2f}")

    # ---- figure -----------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5.8),
                                        gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    yq, okq = finite_eval(fn.f, xq)
    wv = wq[okq] / wq[okq].sum()
    yq = yq[okq]
    ylq = f0 + d0 * (xq - mu)

    if dist.discrete:
        keep = wv > 1e-6
        edges = lattice_edges(yq[keep])
        edges_lin = lattice_edges(ylq[wq > 1e-6]) if s_lin > 0 else edges
    else:
        qlo, qhi = 5e-4, 1 - 5e-4
        lo = min(weighted_quantile(yq, wv, qlo), weighted_quantile(ylq, wq, qlo))
        hi = max(weighted_quantile(yq, wv, qhi), weighted_quantile(ylq, wq, qhi))
        if s_lin > 0:
            lo, hi = min(lo, m_lin - 3.5 * s_lin), max(hi, m_lin + 3.5 * s_lin)
        edges = np.linspace(lo, hi, args.nbins + 1)
        ymin_true = yq.min()
        if lo < ymin_true < hi and ymin_true - lo > (hi - lo) / args.nbins:
            # put a bin edge exactly on the lower boundary of y (e.g. y = x^2 >= 0)
            edges = edges - edges[np.argmin(np.abs(edges - ymin_true))] + ymin_true
        edges_lin = edges

    width = np.diff(edges)
    h_ex = np.histogram(yq, bins=edges, weights=wv)[0] / width
    ax1.hist(y, bins=edges, density=True, histtype="stepfilled", alpha=0.35, color="C0",
             label=rf"toys: $y=f(x)$ ($N={N}$)")
    yy_a = np.linspace(edges[0], edges[-1], 6001)
    pdf_a = analytic_pdf_y(fn, dist, yy_a)
    if pdf_a is not None:
        ax1.plot(yy_a, pdf_a, color="C0", lw=1.8, label=r"exact p.d.f. of $y$ (analytic, change of variables)")
        # cross-check: bin averages of the analytic p.d.f. (midpoint rule, 400 points per bin)
        # vs the numerical quadrature, on bins where f_Y is smooth (no edge or integrable spike)
        sub = (np.arange(400) + 0.5) / 400
        vals = [analytic_pdf_y(fn, dist, lo_ + (hi_ - lo_) * sub) for lo_, hi_ in zip(edges[:-1], edges[1:])]
        h_an = np.array([v.mean() for v in vals])
        smooth = np.array([v.min() > 0 and v.max() / v.min() < 1.5 for v in vals])
        sel = smooth & (h_ex > 0.01 * h_ex.max())
        print(f"analytic p.d.f.: f_Y(y) = {fn.pdf_formula}")
        if sel.any():
            # expected size: one quadrature cell (weight ~ f_X dx) misassigned per bin edge
            print(f"  check vs numerical quadrature ({sel.sum()} smooth bins): max |rel. diff| of bin "
                  f"averages = {np.max(np.abs(h_an[sel] / h_ex[sel] - 1)):.1e}  (quadrature precision ~1e-3)")
    else:
        ax1.stairs(h_ex, edges, color="C0", lw=1.8,
                   label=(r"exact $P(y=f(k))/\Delta y_b$ (Poisson, analytic)" if dist.discrete else
                          r"exact p.d.f. of $y$, bin-averaged: $P(y\in b)/\Delta y_b$ (numerical)"))
    h_lin, _, _ = ax1.hist(y_lin, bins=edges_lin, density=True, histtype="step", lw=1.5, color="C3",
                           label=r"toys: $y_{lin}=f(\mu)+f'(\mu)(x-\mu)$")
    gmax = 0.0
    if s_lin > 0:
        yy = np.linspace(edges[0], edges[-1], 3000)
        g = gauss_pdf((yy - m_lin) / s_lin) / s_lin
        gmax = g.max()
        ax1.plot(yy, g, "C3--", lw=2, label=r"linear prop. as $\mathcal{N}(f(\mu),\,|f'(\mu)|\sigma)$")
    ax1.axvline(m_t, color="k", lw=1.5, label=f"toy mean = {m_t:.4g}")
    ax1.axvline(m_lin, color="C3", lw=1.5, ls=":", label=rf"$f(\mu)$ = {m_lin:.4g}")
    if has_exact:
        ax1.axvline(m_ex, color="C0", lw=1.5, ls=":", label=rf"exact $E[f(x)]$ = {m_ex:.4g}")
    hs = np.sort(h_ex)
    top = max(hs[-2] if hs.size > 1 else hs[-1], gmax, np.max(h_lin) if len(h_lin) else 0)
    ax1.set_ylim(0, 1.85 * top)      # headroom for legend and table
    if has_exact:
        ax1.errorbar(m_ex, 1.10 * top, xerr=s_ex, fmt="o", color="C0", capsize=5, lw=2,
                     label=rf"$E[y]\pm\sigma_y$: cov. {100 * cov['ex'][0]:.1f}%")
    ax1.errorbar(m_lin, 1.03 * top, xerr=s_lin, fmt="o", color="C3", capsize=5, lw=2,
                 label=rf"$f(\mu)\pm\sigma_{{lin}}$: cov. {100 * cov['lin'][0]:.1f}%")
    ax1.set_xlim(min(edges[0], edges_lin[0]), max(edges[-1], edges_lin[-1]))
    ax1.set_xlabel(r"$y = f(x)$")
    ax1.set_ylabel("probability density")
    ax1.set_title(f"{fn.label},   {dist.label},   " + rf"$\sigma/|\mu|={sigma / abs(mu):.2f}$")
    txt = (f"{'':12s}{'mean':>10s}{'sigma':>10s}\n"
           f"{'toys f(x)':12s}{m_t:>10.4g}{s_t:>10.4g}\n"
           + (f"{'exact':12s}{m_ex:>10.4g}{s_ex:>10.4g}\n" if has_exact else
              f"{'exact':12s}{'diverges':>10s}{'diverges':>10s}\n")
           +
           f"{'toys y_lin':12s}{m_tl:>10.4g}{s_tl:>10.4g}\n"
           f"{'linear':12s}{m_lin:>10.4g}{s_lin:>10.4g}")
    txt += f"\n\n{'1-sigma cov.':16s}{'exact':>8s}{'toys':>8s}"
    for k, lab in (("ex", "E+-s_y   on y"), ("lin", "f+-s_lin on y"), ("ylin", "f+-s_lin on ylin"),
                   ("toy", "per-toy f(x_i)+-")):
        if k in cov:
            txt += f"\n{lab:16s}{100 * cov[k][0]:>7.1f}%{100 * cov[k][1]:>7.1f}%"
        else:
            txt += f"\n{lab:16s}{'n/a':>8s}{'n/a':>8s}"
    txt += f"\n(Gaussian: {100 * GAUSS_1SIGMA:.2f}%)"
    if lost_toys > 0:
        txt += f"\ndropped (outside domain): {lost_toys:.1e}"
    ax1.text(0.02, 0.98, txt, transform=ax1.transAxes, ha="left", va="top", family="monospace",
             fontsize=8.5, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax1.legend(fontsize=7.5, loc="upper right")

    # ---- breakdown scan vs sigma/|mu| ---------------------------------------
    scan(fn, dist, args, rng, ax2, ax3)

    fig.tight_layout()
    path = os.path.join(args.outdir, f"{dist.kind}_{fn.name}.png")
    fig.savefig(path, dpi=130)
    print(f"figure: {path}")
    return fig


def scan(fn, dist, args, rng, ax, axc):
    """Bias, width and +-1 sigma coverage w.r.t. linear propagation vs sigma/|mu|."""
    smax = args.scan_max if args.scan_max is not None else fn.scan_max
    r_cur = dist.sigma / abs(dist.mu)
    smax = max(smax, 1.1 * r_cur)

    def dist_at(r):
        if dist.kind == "poisson":
            return InputDist("poisson", 1.0 / r**2, None)       # sigma/mu = 1/sqrt(lambda)
        return InputDist(dist.kind, dist.mu, r * abs(dist.mu))

    def norm(d):
        return abs(fn.deriv(d.mu)) * d.sigma

    if norm(dist) == 0:
        for a in (ax, axc):
            a.text(0.5, 0.5, "f'(mu) = 0: linear propagation gives sigma_y = 0", ha="center",
                   transform=a.transAxes)
        return

    r_fine = np.linspace(0.02, smax, 60)
    bias_ex, ratio_ex, cex_ex, clin_ex, cylin_ex, ctoy_ex = [], [], [], [], [], []
    for r in r_fine:
        d = dist_at(r)
        xq, wq = d.quadrature(40_001)
        y, ok = finite_eval(fn.f, xq)
        yv, wv = y[ok], wq[ok] / wq[ok].sum()
        f0, sl = fn.f(d.mu), norm(d)
        clin_ex.append(coverage_exact(yv, wv, f0 - sl, f0 + sl))
        cylin_ex.append(coverage_exact(f0 + fn.deriv(d.mu) * (xq - d.mu), wq, f0 - sl, f0 + sl))
        cq, okc = per_toy_cover(fn, xq, d.sigma, f0)
        ctoy_ex.append(np.sum(wq[okc & cq]) / np.sum(wq[okc]))
        if pole_in_support(fn, d):
            bias_ex.append(np.nan)
            ratio_ex.append(np.nan)
            cex_ex.append(np.nan)
            continue
        m, s, _ = exact_moments(fn.f, xq, wq)
        bias_ex.append((m - f0) / sl)
        ratio_ex.append(s / sl)
        cex_ex.append(coverage_exact(yv, wv, m - s, m + s))

    n_scan = min(args.ntoys, 20_000)
    r_toy = np.linspace(0.05, smax, 15)
    pts, cpts = [], []
    for r in r_toy:
        d = dist_at(r)
        xs = d.sample(rng, n_scan)
        y, ok = finite_eval(fn.f, xs)
        y = y[ok]
        m, em, s, es = sample_stats(y)
        f0, sl = fn.f(d.mu), norm(d)
        pts.append(((m - f0) / sl, em / sl, s / sl, es / sl))
        if pole_in_support(fn, d):
            ce = (np.nan, np.nan)
        else:
            me, se, _ = exact_moments(fn.f, *d.quadrature(40_001))
            ce = coverage_toys(y, me - se, me + se)
        ct, okt = per_toy_cover(fn, xs, d.sigma, f0)
        pt = float(ct[okt].mean())
        cpts.append((*ce, *coverage_toys(y, f0 - sl, f0 + sl), pt, math.sqrt(pt * (1 - pt) / okt.sum())))
    pts, cpts = np.array(pts), np.array(cpts)

    ax.plot(r_fine, ratio_ex, "C0-", label=r"exact: $\sigma_y/\sigma_{lin}$")
    ax.errorbar(r_toy, pts[:, 2], yerr=pts[:, 3], fmt="o", ms=4, color="C0", label=r"toys: $\sigma_y/\sigma_{lin}$")
    ax.plot(r_fine, bias_ex, "C1-", label=r"exact: $(E[y]-f(\mu))/\sigma_{lin}$")
    ax.errorbar(r_toy, pts[:, 0], yerr=pts[:, 1], fmt="s", ms=4, color="C1", label="toys: bias")
    big = np.nanmax(np.abs(np.concatenate([ratio_ex, bias_ex, pts[:, 0], pts[:, 2]])))
    if big > 10:     # e.g. exp (log-normal) or 1/x: switch to a symmetric log axis
        ax.set_yscale("symlog", linthresh=1.0)
    ax.axhline(1, color="C0", ls="--", lw=1)
    ax.axhline(0, color="C1", ls="--", lw=1, label="linear propagation")
    ax.axvline(r_cur, color="0.5", ls=":", lw=1)
    xl = r"relative uncertainty $\sigma/|\mu|$"
    if dist.kind == "poisson":
        xl += r"   ($=1/\sqrt{\lambda}$, $\lambda$ varied)"
    ax.set_xlabel(xl)
    ax.set_ylabel(r"w.r.t. linear propagation, $\sigma_{lin}=|f'(\mu)|\sigma$")
    ttl = f"Breakdown of linear propagation ({n_scan} toys/point)"
    if np.all(np.isnan(ratio_ex)):
        ttl += "\nexact moments do not exist (pole inside support): toys only, not convergent"
    elif np.any(np.isnan(ratio_ex)):
        ttl += "\nno exact curve where the pole enters the support of x"
    ax.set_title(ttl, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # ---- coverage panel -----------------------------------------------------
    axc.plot(r_fine, 100 * np.array(cex_ex), "C0-", label=r"exact: $y\in[E[y]\pm\sigma_y]$")
    axc.errorbar(r_toy, 100 * cpts[:, 0], yerr=100 * cpts[:, 1], fmt="o", ms=4, color="C0", label="toys")
    axc.plot(r_fine, 100 * np.array(clin_ex), "C3-", label=r"exact: $y\in[f(\mu)\pm\sigma_{lin}]$")
    axc.errorbar(r_toy, 100 * cpts[:, 2], yerr=100 * cpts[:, 3], fmt="s", ms=4, color="C3", label="toys")
    axc.plot(r_fine, 100 * np.array(ctoy_ex), "C2-",
             label=r"exact: $f(x_i)\pm|f'(x_i)|\sigma\ni f(\mu)$ (per toy)")
    axc.errorbar(r_toy, 100 * cpts[:, 4], yerr=100 * cpts[:, 5], fmt="^", ms=4, color="C2", label="toys")
    axc.plot(r_fine, 100 * np.array(cylin_ex), color="0.4", ls="-.",
             label=r"$y_{lin}\in[f(\mu)\pm\sigma_{lin}]$ $=P(|x-\mu|\leq\sigma)$")
    axc.axhline(100 * GAUSS_1SIGMA, color="k", ls="--", lw=1, label="Gaussian 68.27%")
    axc.axvline(r_cur, color="0.5", ls=":", lw=1)
    axc.set_xlabel(xl)
    axc.set_ylabel(r"coverage of the $\pm1\sigma$ interval [%]")
    axc.set_title(r"$\pm1\sigma$ coverage on the true $y=f(x)$", fontsize=10)
    axc.legend(fontsize=8, loc="best")
    axc.grid(alpha=0.3)


# =========================================================================== #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--funcs", nargs="+", default=DEFAULT_FUNCS,
                   help=f"built-in names or numpy expressions in x (default: {' '.join(DEFAULT_FUNCS)})")
    p.add_argument("--dist", choices=["gauss", "uniform", "poisson"], default="gauss")
    p.add_argument("--mu", type=float, default=5.0, help="mean of x (Poisson: lambda)")
    p.add_argument("--sigma", type=float, default=1.0, help="std of x (ignored for Poisson)")
    p.add_argument("--ntoys", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nbins", type=int, default=100)
    p.add_argument("--scan-max", type=float, default=None,
                   help="max sigma/|mu| in the scan (default: per-function value)")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
                   help="output directory (default: plots/ next to this script)")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--list-funcs", action="store_true")
    args = p.parse_args()

    if args.list_funcs:
        for k, fn in FUNCS.items():
            print(f"{k:6s} {fn.label:14s} {fn.note}")
        return

    if args.dist == "poisson" and args.sigma != p.get_default("sigma"):
        print(f"note: --sigma ignored for Poisson, sigma = sqrt({args.mu:g})")
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dist = InputDist(args.dist, args.mu, args.sigma)
    with np.errstate(all="ignore"):     # overflows (e.g. exp at large x) are handled as non-finite
        for spec in args.funcs:
            run_one(make_func(spec), dist, args, rng)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
