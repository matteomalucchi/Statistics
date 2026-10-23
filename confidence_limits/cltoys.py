"""
cltoys: confidence intervals, upper limits and exclusion with toys and asymptotics.

A small, self-contained library (numpy + scipy) used by the scripts 01-05 and by
the notebook in this directory. Everything is vectorised over toys, so 10^4-10^5
pseudo-experiments run in seconds.

Contents
--------
1. Classical (Neyman) intervals for Gaussian and Poisson observables
   - poisson_interval, gaussian_interval, exact coverage
2. Feldman-Cousins (likelihood-ratio ordering)
   - fc_poisson_belt / fc_poisson_interval, fc_gaussian_belt / fc_gaussian_interval
3. Counting experiment limits: CL_{s+b}, CLs, Bayesian (flat prior)
4. A binned Poisson likelihood with one nuisance parameter (class BinnedModel)
      nu_i(mu, theta) = mu * s_i + b_i * g(theta),   theta ~ N(a, 1)
   with g(theta) = kappa**theta (log-normal, Combine "lnN") or 1 + delta*theta
   (Gaussian), toy generation (frequentist toys, randomised global observable a),
   Asimov data, vectorised maximum-likelihood fits and the LHC test statistics
   t_mu, q_mu, q~_mu (qtilde) and q_0.
5. CLs from toys and from the asymptotic formulae of Cowan, Cranmer, Gross,
   Vitells, EPJC 71 (2011) 1554 [arXiv:1007.1727] (STAMET/papers/1007.1727.pdf),
   expected limits and the +-1 sigma, +-2 sigma bands.

Notation follows the STAMET book notebook confidenceIntervals.ipynb (c2, c7,
c14, c16-c20) and 1007.1727: mu = signal strength (POI), theta = nuisance,
lambda(mu) = L(mu, theta_hathat) / L(mu_hat, theta_hat).
"""
from __future__ import annotations

import numpy as np
from scipy import stats, optimize, integrate

Phi = stats.norm.cdf          # standard normal CDF
Phi_inv = stats.norm.ppf      # its inverse (quantile)


def z_to_p(z):
    """One-sided p-value for a significance Z: p = 1 - Phi(Z)."""
    return stats.norm.sf(z)


def p_to_z(p):
    """Significance for a one-sided p-value: Z = Phi^{-1}(1 - p)."""
    return stats.norm.isf(p)


# =============================================================================
# 1. Classical Neyman intervals
# =============================================================================
def gaussian_interval(x, sigma=1.0, cl=0.68269, kind="central"):
    """Neyman interval for the mean of a Gaussian with known sigma.

    kind = 'central' : [x - z sigma, x + z sigma], z = Phi^{-1}(1 - (1-cl)/2)
    kind = 'upper'   : (-inf, x + sigma Phi^{-1}(cl)]
    kind = 'lower'   : [x - sigma Phi^{-1}(cl), +inf)
    (book confidenceIntervals c2, Cowan 9.3)
    """
    x = np.asarray(x, float)
    if kind == "central":
        z = Phi_inv(1 - (1 - cl) / 2)
        return x - z * sigma, x + z * sigma
    if kind == "upper":
        return np.full_like(x, -np.inf), x + sigma * Phi_inv(cl)
    if kind == "lower":
        return x - sigma * Phi_inv(cl), np.full_like(x, np.inf)
    raise ValueError(kind)


def poisson_interval(n, cl=0.68269, kind="central", b=0.0):
    """Classical (Neyman) interval for the signal mean s of a Poisson count
    n ~ Pois(s + b) with known background b. Exact (Garwood) construction:

      lower:  P(N >= n | s_lo + b) = alpha  ->  s_lo + b = chi2_{alpha}(2n) / 2
      upper:  P(N <= n | s_up + b) = beta   ->  s_up + b = chi2_{1-beta}(2n+2) / 2

    kind = 'central' uses alpha = beta = (1-cl)/2; 'upper' uses beta = 1-cl.
    The limits are NOT clipped at s = 0: with b > 0 they can be negative, and the
    interval can be empty (book c7: b = 3, n = 0). Returns (s_lo, s_up).
    n = 0 and b = 0 gives s_up = -ln(1-cl) (= 2.996 at 95% CL, book c2).
    """
    n = np.asarray(n, float)
    if kind == "central":
        a = (1 - cl) / 2
        lo = np.where(n > 0, 0.5 * stats.chi2.ppf(a, 2 * n), 0.0) - b
        up = 0.5 * stats.chi2.ppf(1 - a, 2 * n + 2) - b
    elif kind == "upper":
        lo = np.full_like(n, -b)
        up = 0.5 * stats.chi2.ppf(cl, 2 * n + 2) - b
    else:
        raise ValueError(kind)
    return lo, up


def poisson_exact_coverage(lam_true, interval_fn, nmax=None):
    """Exact coverage P(lam_true in [lo(n), up(n)]) = sum_n Pois(n|lam) 1[...].

    interval_fn(n_array) -> (lo, up) for the *total* mean lambda.
    """
    lam_true = np.atleast_1d(np.asarray(lam_true, float))
    if nmax is None:
        nmax = int(lam_true.max() + 12 * np.sqrt(lam_true.max() + 1) + 20)
    n = np.arange(nmax + 1)
    lo, up = interval_fn(n)
    cov = []
    for lam in lam_true:
        inside = (lo <= lam) & (lam <= up)
        cov.append(stats.poisson.pmf(n[inside], lam).sum())
    return np.array(cov)


# =============================================================================
# 2. Feldman-Cousins
# =============================================================================
def fc_poisson_acceptance(mu, b, cl=0.90, nmax=None):
    """FC acceptance region {n} for signal mean mu and known background b.

    Rank R(n) = P(n | mu + b) / P(n | mu_best + b), mu_best = max(0, n - b);
    add n in decreasing R until sum P >= cl (book c7; FC paper Sec. IV).
    Returns (n_min, n_max, accumulated probability).
    """
    if nmax is None:
        nmax = int(mu + b + 10 * np.sqrt(mu + b + 1) + 20)
    n = np.arange(nmax + 1)
    p = stats.poisson.pmf(n, mu + b)
    mu_best = np.maximum(0.0, n - b)
    r = p / stats.poisson.pmf(n, mu_best + b)
    order = np.argsort(-r, kind="stable")
    csum = np.cumsum(p[order])
    k = np.searchsorted(csum, cl) + 1           # first index where csum >= cl
    acc = n[order[:k]]
    return acc.min(), acc.max(), csum[k - 1]


def fc_poisson_belt(b, cl=0.90, mu_grid=None):
    """FC confidence belt: arrays (mu_grid, n_min(mu), n_max(mu))."""
    if mu_grid is None:
        mu_grid = np.arange(0.0, 20.0 + 1e-9, 0.005)
    lo = np.empty(mu_grid.size, int)
    hi = np.empty(mu_grid.size, int)
    for i, mu in enumerate(mu_grid):
        lo[i], hi[i], _ = fc_poisson_acceptance(mu, b, cl)
    return mu_grid, lo, hi


def fc_poisson_interval(n_obs, b, cl=0.90, mu_grid=None, belt=None,
                        monotone_b=False, db_max=2.0, db_step=0.01):
    """FC interval [mu_lo, mu_up] for an observed count (read the belt vertically).
    Resolution is the mu_grid step (default 0.005). If the vertical line meets
    several disjoint segments, the bottom- and topmost are taken (FC paper p.6).

    monotone_b=True applies FC's second correction (paper p.6, right column):
    the upper end point is forced to be non-increasing in b by taking
    mu_up(n, b) = max_{b' in [b, b + db_max]} mu_up_raw(n, b'). Without it the
    raw construction gives e.g. [0, 0.95] for b = 3, n = 0 instead of the
    tabulated [0, 1.08]; all other b = 3 entries agree. (Slow: ~10 s.)
    """
    mu, lo, hi = belt if belt is not None else fc_poisson_belt(b, cl, mu_grid)
    inside = (lo <= n_obs) & (n_obs <= hi)
    if not inside.any():
        return np.nan, np.nan
    mu_lo, mu_up = mu[inside].min(), mu[inside].max()
    if monotone_b:
        step = mu[1] - mu[0]
        grid = np.arange(0.0, mu_up + 3.0, step)
        for bp in np.arange(b + db_step, b + db_max + 1e-9, db_step):
            m2, lo2, hi2 = fc_poisson_belt(bp, cl, grid)
            ins = (lo2 <= n_obs) & (n_obs <= hi2)
            if ins.any():
                mu_up = max(mu_up, m2[ins].max())
    return mu_lo, mu_up


def fc_gaussian_belt(cl=0.90, mu_grid=None, x_grid=None):
    """FC belt for x ~ N(mu, 1) with the physical constraint mu >= 0.

    R(x) = phi(x - mu) / phi(x - mu_best), mu_best = max(0, x)  (book c8).
    For each mu the acceptance region is {x : R(x) >= k} with P = cl.
    Returns (mu_grid, x1(mu), x2(mu)).
    """
    if mu_grid is None:
        mu_grid = np.linspace(0, 6, 601)
    if x_grid is None:
        x_grid = np.linspace(-8, 14, 22001)
    dx = x_grid[1] - x_grid[0]
    mu_best = np.maximum(0.0, x_grid)
    x1 = np.empty(mu_grid.size)
    x2 = np.empty(mu_grid.size)
    for i, mu in enumerate(mu_grid):
        pdf = stats.norm.pdf(x_grid, mu)
        logr = -0.5 * (x_grid - mu) ** 2 + 0.5 * (x_grid - mu_best) ** 2
        order = np.argsort(-logr, kind="stable")
        csum = np.cumsum(pdf[order]) * dx
        k = np.searchsorted(csum, cl) + 1
        acc = x_grid[order[:k]]
        x1[i], x2[i] = acc.min(), acc.max()
    return mu_grid, x1, x2


def fc_gaussian_interval(x_obs, belt):
    """Read the FC Gaussian belt vertically at x_obs."""
    mu, x1, x2 = belt
    inside = (x1 <= x_obs) & (x_obs <= x2)
    if not inside.any():          # x_obs outside the tabulated x range
        return np.nan, np.nan
    return mu[inside].min(), mu[inside].max()


def flipflop_interval(x, cl=0.90):
    """The 'flip-flop' policy of FC paper Sec. III / book c6 (sigma = 1):
    if x >= 3 quote a central cl interval, otherwise a cl upper limit, and if
    x < 0 pretend x = 0 for the upper limit."""
    x = np.asarray(x, float)
    zc = Phi_inv(1 - (1 - cl) / 2)
    zu = Phi_inv(cl)
    lo = np.where(x >= 3, x - zc, 0.0)
    up = np.where(x >= 3, x + zc, np.maximum(x, 0.0) + zu)
    return lo, up


def gaussian_interval_coverage(mu_true, interval_fn, x_grid=None):
    """Coverage of an interval prescription for x ~ N(mu, 1), by integration."""
    if x_grid is None:
        x_grid = np.linspace(-10, 20, 30001)
    dx = x_grid[1] - x_grid[0]
    lo, up = interval_fn(x_grid)
    out = []
    for mu in np.atleast_1d(mu_true):
        inside = (lo <= mu) & (mu <= up)
        out.append(stats.norm.pdf(x_grid[inside], mu).sum() * dx)
    return np.array(out)


# =============================================================================
# 3. Counting experiment: CL_{s+b}, CL_b, CLs, Bayesian
# =============================================================================
def counting_cls(n_obs, s, b):
    """For a single-bin counting experiment with known b and test statistic n
    (small n = background-like; equivalent to the LEP -2 ln Q for one bin):

      CL_{s+b} = P(N <= n_obs | s + b)   (= p_{s+b}, the p-value of s+b)
      CL_b     = P(N <= n_obs | b)       (= 1 - p_b)
      CLs      = CL_{s+b} / CL_b          (book c14; L9 p.45)
    """
    clsb = stats.poisson.cdf(n_obs, s + b)
    clb = stats.poisson.cdf(n_obs, b)
    return clsb, clb, clsb / clb


def counting_upper_limit(n_obs, b, cl=0.95, method="CLs"):
    """Upper limit on s for n_obs ~ Pois(s + b), known b.

    method = 'CLsb' : classical Neyman upper limit (may be negative)
    method = 'CLs'  : CL_{s+b}(s_up) / CL_b = 1 - cl   (always > 0)
    method = 'bayes': flat prior pi(s) = 1 for s >= 0, by numerical integration.
    """
    alpha = 1 - cl
    if method == "CLsb":
        return poisson_interval(n_obs, cl, "upper", b)[1]
    if method == "CLs":
        clb = stats.poisson.cdf(n_obs, b)
        # P(N <= n | lam) = alpha * CL_b  -> lam = chi2_{1 - alpha CL_b}(2n+2)/2
        return 0.5 * stats.chi2.ppf(1 - alpha * clb, 2 * n_obs + 2) - b
    if method == "bayes":
        like = lambda s: stats.poisson.pmf(n_obs, s + b)
        norm = integrate.quad(like, 0, np.inf, limit=200)[0]
        f = lambda su: integrate.quad(like, 0, su, limit=200)[0] / norm - cl
        return optimize.brentq(f, 1e-9, n_obs + b + 50 * np.sqrt(n_obs + b + 1) + 50)
    raise ValueError(method)


# =============================================================================
# 4. Binned likelihood with one nuisance parameter
# =============================================================================
class BinnedModel:
    """Binned Poisson model  nu_i = mu s_i + b_i g(theta)  with a unit-Gaussian
    constraint on theta from an auxiliary measurement a ~ N(theta, 1)
    (Combine's "global observable", P p.4-5).

        -ln L(mu, theta) = sum_i [nu_i - n_i ln nu_i] + (theta - a)^2 / 2 + const

    nuisance = None            : no nuisance (theta fixed to 0, b known)
    nuisance = ('lnN', kappa)  : g = kappa**theta   (log-normal, Combine lnN)
    nuisance = ('gauss', d)    : g = 1 + d theta    (Gaussian, truncated at g > 0)

    All methods are vectorised: n has shape (T, B) for T toys, a has shape (T,).
    """

    def __init__(self, s, b, nuisance=None):
        self.s = np.atleast_1d(np.asarray(s, float))
        self.b = np.atleast_1d(np.asarray(b, float))
        self.nuisance = nuisance
        self.has_nuis = nuisance is not None

    # ---- model -----------------------------------------------------------
    def _g(self, th):
        if not self.has_nuis:
            return np.ones_like(th), np.zeros_like(th), np.zeros_like(th)
        kind, par = self.nuisance
        if kind == "lnN":
            lk = np.log(par)
            g = np.exp(th * lk)
            return g, g * lk, g * lk * lk
        if kind == "gauss":
            return 1 + par * th, np.full_like(th, par), np.zeros_like(th)
        raise ValueError(kind)

    def expected(self, mu, theta=0.0):
        mu = np.asarray(mu, float)[..., None]
        g = self._g(np.asarray(theta, float))[0][..., None]
        return mu * self.s + self.b * g

    def nll(self, mu, th, n, a):
        nu = self.expected(mu, th)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(n > 0, n * np.log(nu), 0.0)
        out = np.sum(nu - t, axis=-1)
        if self.has_nuis:
            out = out + 0.5 * (th - a) ** 2
        return np.where(np.all(nu > 0, axis=-1), out, np.inf)

    # ---- data --------------------------------------------------------------
    def generate(self, mu, theta, ntoys, rng):
        """Frequentist toys: n_i ~ Pois(nu_i(mu, theta)), a ~ N(theta, 1).
        theta is the value used for generation (in the LHC recipe, the value
        fitted to the data at fixed mu, i.e. theta_hathat(mu))."""
        nu = self.expected(np.full(ntoys, mu), np.full(ntoys, theta))
        n = rng.poisson(nu).astype(float)
        a = rng.normal(theta, 1.0, ntoys) if self.has_nuis else np.zeros(ntoys)
        return n, a

    def asimov(self, mu, theta=0.0):
        """Asimov data: n_i = E[n_i] (not integer), a = theta (1007.1727 Sec. 3.2)."""
        return self.expected(np.array([mu]), np.array([theta])), np.array([theta])

    # ---- fits --------------------------------------------------------------
    def _derivs(self, mu, th, n, a, need_mu=True):
        g, g1, g2 = self._g(th)
        nu = mu[:, None] * self.s + self.b * g[:, None]
        w = n / nu
        w2 = n / nu ** 2
        out = {}
        if need_mu:
            out["gm"] = np.sum(self.s * (1 - w), axis=1)
            out["hmm"] = np.sum(self.s ** 2 * w2, axis=1)
        if self.has_nuis:
            bg1 = self.b * g1[:, None]
            out["gt"] = np.sum(bg1 * (1 - w), axis=1) + (th - a)
            out["htt"] = (np.sum(bg1 ** 2 * w2, axis=1)
                          + np.sum(self.b * g2[:, None] * (1 - w), axis=1) + 1.0)
            if need_mu:
                out["hmt"] = np.sum(self.s * bg1 * w2, axis=1)
        return out

    def profile_theta(self, mu, n, a, th0=None, iters=60):
        """Conditional fit theta_hathat(mu) for fixed mu (scalar or per toy).
        Returns (theta_hathat, nll)."""
        n = np.atleast_2d(n)
        T = n.shape[0]
        mu = np.broadcast_to(np.asarray(mu, float), (T,)).copy()
        a = np.broadcast_to(np.asarray(a, float), (T,)).copy()
        if not self.has_nuis:
            th = np.zeros(T)
            return th, self.nll(mu, th, n, a)
        th = a.copy() if th0 is None else np.array(th0, float)
        f = self.nll(mu, th, n, a)
        bad = ~np.isfinite(f)                   # start point infeasible
        if bad.any():
            th[bad] = 0.0 if self.nuisance[0] == "lnN" else 0.0
            f = self.nll(mu, th, n, a)
        for _ in range(iters):
            d = self._derivs(mu, th, n, a, need_mu=False)
            h = np.where(d["htt"] > 1e-12, d["htt"], 1.0)
            step = np.clip(-d["gt"] / h, -2.0, 2.0)   # trust region: |d theta| <= 2
            t = np.ones(T)
            for _ in range(40):
                f_new = self.nll(mu, th + t * step, n, a)
                ok = np.isfinite(f_new) & (f_new <= f + 1e-4 * t * step * d["gt"] + 1e-12)
                if ok.all():
                    break
                t = np.where(ok, t, 0.5 * t)
            f_new = self.nll(mu, th + t * step, n, a)
            acc = np.isfinite(f_new) & (f_new <= f + 1e-12)
            th = np.where(acc, th + t * step, th)
            f = np.where(acc, f_new, f)
            if np.all(np.abs(step * d["gt"]) < 1e-12):
                break
        return th, f

    def fit(self, n, a, mu0=None, iters=100):
        """Unconditional ML fit (mu_hat, theta_hat, nll_min), mu unbounded
        (it may be negative as long as all nu_i > 0; book c16)."""
        n = np.atleast_2d(n)
        T = n.shape[0]
        a = np.broadcast_to(np.asarray(a, float), (T,)).copy()
        # starting point: background-subtracted signal estimate
        if mu0 is None:
            mu = np.sum(n - self.b, axis=1) / np.sum(self.s)
            mu = np.maximum(mu, -0.9 * np.min(self.b / self.s))
        else:
            mu = np.broadcast_to(np.asarray(mu0, float), (T,)).copy()
        th = a.copy() if self.has_nuis else np.zeros(T)
        f = self.nll(mu, th, n, a)
        bad = ~np.isfinite(f)
        if bad.any():
            th[bad] = 0.0
            mu[bad] = 0.0
            f = self.nll(mu, th, n, a)
        for _ in range(iters):
            d = self._derivs(mu, th, n, a)
            if self.has_nuis:
                hmm, hmt, htt = d["hmm"], d["hmt"], d["htt"]
                # make the 2x2 Hessian positive definite (Levenberg shift)
                tr, det = hmm + htt, hmm * htt - hmt ** 2
                lmin = 0.5 * (tr - np.sqrt(np.maximum(tr ** 2 - 4 * det, 0)))
                lam = np.where(lmin > 1e-10 * (1 + np.abs(tr)), 0.0,
                               -lmin + 1e-8 * (1 + np.abs(tr)))
                hmm, htt = hmm + lam, htt + lam
                det = hmm * htt - hmt ** 2
                sm = -(htt * d["gm"] - hmt * d["gt"]) / det
                st = -(-hmt * d["gm"] + hmm * d["gt"]) / det
                dirder = sm * d["gm"] + st * d["gt"]
            else:
                h = np.where(d["hmm"] > 1e-12, d["hmm"], 1e-12)
                sm, st = -d["gm"] / h, np.zeros(T)
                dirder = sm * d["gm"]
            # trust region on theta (|d theta| <= 2) keeps kappa**theta finite
            t = np.minimum(1.0, 2.0 / np.maximum(np.abs(st), 1e-300))
            for _ in range(50):
                f_new = self.nll(mu + t * sm, th + t * st, n, a)
                ok = np.isfinite(f_new) & (f_new <= f + 1e-4 * t * dirder + 1e-12)
                if ok.all():
                    break
                t = np.where(ok, t, 0.5 * t)
            f_new = self.nll(mu + t * sm, th + t * st, n, a)
            acc = np.isfinite(f_new) & (f_new <= f + 1e-12)
            mu = np.where(acc, mu + t * sm, mu)
            th = np.where(acc, th + t * st, th)
            f = np.where(acc, f_new, f)
            if np.all(np.abs(dirder) < 1e-12):
                break
        return mu, th, f

    # ---- test statistics ----------------------------------------------------
    def test_statistics(self, mu, n, a, fit=None):
        """Return a dict with t_mu, q_mu, qtilde_mu (and mu_hat) for tested mu.

        t_mu      = -2 ln lambda(mu)                         (two-sided)
        q_mu      = t_mu if mu_hat <= mu else 0              (book c20)
        qtilde_mu = -2 ln[L(mu)/L(0)]      if mu_hat < 0
                    -2 ln[L(mu)/L(mu_hat)] if 0 <= mu_hat <= mu
                    0                      if mu_hat > mu     (C:statistical_tests L173)
        """
        n = np.atleast_2d(n)
        mu_hat, th_hat, f_hat = fit if fit is not None else self.fit(n, a)
        _, f_mu = self.profile_theta(mu, n, a)
        t = np.maximum(2 * (f_mu - f_hat), 0.0)
        q = np.where(mu_hat <= mu, t, 0.0)
        neg = mu_hat < 0
        qt = q.copy()
        if neg.any():
            _, f_0 = self.profile_theta(0.0, n[neg], np.asarray(a)[neg] if np.ndim(a) else a)
            qt[neg] = np.maximum(2 * (f_mu[neg] - f_0), 0.0)
        return {"t": t, "q": q, "qtilde": qt, "mu_hat": mu_hat}

    def q0(self, n, a, fit=None):
        """Discovery statistic q0 = -2 ln lambda(0) if mu_hat >= 0, else 0 (book c17)."""
        n = np.atleast_2d(n)
        mu_hat, _, f_hat = fit if fit is not None else self.fit(n, a)
        _, f0 = self.profile_theta(0.0, n, a)
        return np.where(mu_hat >= 0, np.maximum(2 * (f0 - f_hat), 0.0), 0.0), mu_hat


# =============================================================================
# 5. CLs: asymptotic formulae and toys
# =============================================================================
def sigma_asimov(model, mu, theta_b=0.0):
    """sigma of mu_hat from the background-only Asimov data set:
    sigma_A^2 = mu^2 / qtilde_{mu,A}  (1007.1727 eq. 30-32; book c19)."""
    n_A, a_A = model.asimov(0.0, theta_b)
    qA = model.test_statistics(mu, n_A, a_A)["qtilde"][0]
    return mu / np.sqrt(max(qA, 1e-300))


def asymptotic_cls(qt, mu, sigma):
    """Asymptotic p_mu, 1 - p_b and CLs for the observed qtilde_mu
    (1007.1727 eqs. 65-66, sigma from the Asimov data set).

      F(q~|mu') = Phi(sqrt(q~) - (mu-mu')/sigma)                 q~ <= mu^2/sigma^2
                = Phi((q~ - (mu^2 - 2 mu mu')/sigma^2)/(2mu/sigma)) q~ >  mu^2/sigma^2
      p_mu = 1 - F(q~|mu),   1 - p_b = 1 - F(q~|0),   CLs = p_mu / (1 - p_b)
    """
    qt = np.asarray(qt, float)
    r = (mu / sigma) ** 2
    sq = np.sqrt(qt)
    small = qt <= r
    with np.errstate(divide="ignore", invalid="ignore"):
        p_mu = np.where(small, stats.norm.sf(sq), stats.norm.sf((qt + r) / (2 * mu / sigma)))
        one_m_pb = np.where(small, stats.norm.sf(sq - mu / sigma),
                            stats.norm.sf((qt - r) / (2 * mu / sigma)))
    return p_mu / one_m_pb, p_mu, one_m_pb


def asymptotic_expected_limit(model, cl=0.95, n_sigma=0.0, theta_b=0.0, iters=50):
    """Expected CLs limit at quantile N (N = 0 median, +-1, +-2 bands):
        mu_up(N) = sigma_A(mu_up) * [Phi^{-1}(1 - alpha Phi(N)) + N]
    (1007.1727 eq. 89; solved by fixed-point iteration since sigma_A depends on mu)."""
    alpha = 1 - cl
    k = Phi_inv(1 - alpha * Phi(n_sigma)) + n_sigma
    mu = 1.0
    for _ in range(iters):
        mu_new = sigma_asimov(model, mu, theta_b) * k
        if abs(mu_new - mu) < 1e-6 * (1 + mu):
            mu = mu_new
            break
        mu = mu_new
    return mu


def asymptotic_observed_limit(model, n_obs, a_obs=0.0, cl=0.95, mu_max=None):
    """Observed asymptotic CLs limit: solve CLs(mu) = 1 - cl with qtilde_obs(mu)
    from the data and sigma_A(mu) from the background-only Asimov data."""
    alpha = 1 - cl
    n_obs = np.atleast_2d(n_obs)
    fit = model.fit(n_obs, a_obs)

    def f(mu):
        qt = model.test_statistics(mu, n_obs, a_obs, fit=fit)["qtilde"][0]
        return np.log(asymptotic_cls(qt, mu, sigma_asimov(model, mu))[0]) - np.log(alpha)

    if mu_max is None:
        mu_max = 10 * asymptotic_expected_limit(model, cl, 2.0) + 10 * max(fit[0][0], 0) + 1
    return optimize.brentq(f, 1e-6 * mu_max, mu_max, xtol=1e-6)


def toy_cls_point(model, n_obs, a_obs, mu, ntoys, rng, stat="qtilde"):
    """Toy-based CLs at one tested mu (LHC recipe, Combine HybridNew --LHCmode
    LHC-limits): nuisances in the toys are set to their conditional fit to the
    data, theta_hathat(mu) for s+b toys and theta_hathat(0) for b-only toys; the
    global observable is randomised around them.

    Returns dict with q_obs, p_mu, 1-p_b, CLs (with binomial errors) and the toy
    arrays q_sb, q_b.
    """
    n_obs = np.atleast_2d(n_obs)
    q_obs = model.test_statistics(mu, n_obs, a_obs)[stat][0]
    th_mu, _ = model.profile_theta(mu, n_obs, a_obs)
    th_0, _ = model.profile_theta(0.0, n_obs, a_obs)
    n_sb, a_sb = model.generate(mu, th_mu[0], ntoys, rng)
    n_b, a_b = model.generate(0.0, th_0[0], ntoys, rng)
    q_sb = model.test_statistics(mu, n_sb, a_sb)[stat]
    q_b = model.test_statistics(mu, n_b, a_b)[stat]
    p_mu = np.mean(q_sb >= q_obs)
    omp_b = np.mean(q_b >= q_obs)
    e_mu = np.sqrt(p_mu * (1 - p_mu) / ntoys)
    e_b = np.sqrt(omp_b * (1 - omp_b) / ntoys)
    cls = p_mu / omp_b if omp_b > 0 else np.nan
    e_cls = cls * np.hypot(e_mu / max(p_mu, 1e-300), e_b / max(omp_b, 1e-300))
    return dict(mu=mu, q_obs=q_obs, p_mu=p_mu, one_minus_pb=omp_b, CLs=cls,
                CLs_err=e_cls, q_sb=q_sb, q_b=q_b)


def crossing(mu_grid, cls_vals, alpha=0.05):
    """First crossing of CLs(mu) = alpha on an increasing mu grid (log-linear
    interpolation). Returns nan if there is no crossing."""
    mu_grid = np.asarray(mu_grid)
    y = np.log(np.clip(np.asarray(cls_vals, float), 1e-300, None)) - np.log(alpha)
    idx = np.where((y[:-1] > 0) & (y[1:] <= 0))[0]
    if idx.size == 0:
        return np.nan
    i = idx[0]
    return mu_grid[i] + (mu_grid[i + 1] - mu_grid[i]) * y[i] / (y[i] - y[i + 1])


def toy_limit_scan(model, n_obs, a_obs, mu_grid, ntoys, rng, cl=0.95,
                   quantiles=(0.025, 0.16, 0.5, 0.84, 0.975)):
    """Toy CLs on a grid of mu: observed limit and expected limits from the same
    toys ('expected from grid', as Combine HybridNew does).

    For the expected limit at quantile k of the limit distribution, the observed
    q is replaced by the (1-k) quantile of f(q~_mu | 0): a larger q~ means a
    stronger exclusion, i.e. a lower limit.
    """
    alpha = 1 - cl
    pts = [toy_cls_point(model, n_obs, a_obs, mu, ntoys, rng) for mu in mu_grid]
    cls_obs = np.array([p["CLs"] for p in pts])
    exp = {}
    for k in quantiles:
        vals = []
        for p in pts:
            qk = np.quantile(p["q_b"], 1 - k)
            num = np.mean(p["q_sb"] >= qk)
            den = np.mean(p["q_b"] >= qk)
            vals.append(num / den if den > 0 else np.nan)
        exp[k] = crossing(mu_grid, vals, alpha)
    return dict(points=pts, CLs_obs=cls_obs, obs=crossing(mu_grid, cls_obs, alpha),
                expected=exp)


def per_toy_asymptotic_limits(model, n, a, cl=0.95, mu_hi=None, iters=40):
    """Asymptotic CLs limit for every toy (vectorised bisection on mu); used to
    build the distribution of upper limits under the background-only hypothesis."""
    alpha = 1 - cl
    n = np.atleast_2d(n)
    T = n.shape[0]
    a = np.broadcast_to(np.asarray(a, float), (T,))
    fit = model.fit(n, a)
    if mu_hi is None:
        mu_hi = 6 * asymptotic_expected_limit(model, cl, 2.0)
    lo = np.full(T, 1e-6 * mu_hi)
    hi = np.full(T, mu_hi)
    # cache sigma_A on a grid (it depends on mu only, not on the toy)
    mus = np.linspace(1e-3 * mu_hi, mu_hi, 60)
    sig = np.array([sigma_asimov(model, m) for m in mus])
    sigma_of = lambda m: np.interp(m, mus, sig)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, f_mu = model.profile_theta(mid, n, a)
        mu_hat, _, f_hat = fit
        _, f_0 = model.profile_theta(0.0, n, a)
        t = np.maximum(2 * (f_mu - f_hat), 0)
        t0 = np.maximum(2 * (f_mu - f_0), 0)
        qt = np.where(mu_hat > mid, 0.0, np.where(mu_hat < 0, t0, t))
        cls = asymptotic_cls(qt, mid, sigma_of(mid))[0]
        excluded = cls < alpha
        hi = np.where(excluded, mid, hi)
        lo = np.where(excluded, lo, mid)
    return 0.5 * (lo + hi)


# =============================================================================
# Helpers: a "bump hunt" model (Gaussian peak on a falling background)
# =============================================================================
def bump_model(mass=125.0, width=2.0, s_tot=20.0, b_tot=1000.0, slope=0.03,
               lo=100.0, hi=180.0, nbins=40, nuisance=("lnN", 1.10)):
    """Binned m_X spectrum: exponential background exp(-slope m) normalised to
    b_tot in [lo, hi], Gaussian signal (mean mass, sigma width) normalised to
    s_tot events for mu = 1. Nuisance: background normalisation (default lnN
    10%). Returns (BinnedModel, bin_edges)."""
    edges = np.linspace(lo, hi, nbins + 1)
    cb = np.exp(-slope * edges)
    b = b_tot * (cb[:-1] - cb[1:]) / (cb[0] - cb[-1])
    cs = stats.norm.cdf(edges, mass, width)
    s = s_tot * np.diff(cs)
    s = np.maximum(s, 1e-12)
    return BinnedModel(s, b, nuisance), edges


# =============================================================================
# Plot style shared by the scripts
# =============================================================================
# Fixed categorical order (always assigned in this order, never cycled); line
# styles carry identity too, so the plots survive greyscale printing.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BRAZIL_1S, BRAZIL_2S = "#00cc00", "#ffcc00"     # HEP convention for +-1/2 sigma bands


def setup_style(agg=True):
    """Common matplotlib style; agg=False inside a notebook (inline backend)."""
    import matplotlib
    if agg:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "legend.frameon": False, "font.size": 10,
    })
    return plt


def qtilde_cdf_asym(q, mu, mu_prime, sigma):
    """Asymptotic CDF F(q~_mu | mu') (1007.1727 eq. 65), sigma from Asimov."""
    q = np.asarray(q, float)
    r = (mu / sigma) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(q <= r, Phi(np.sqrt(np.maximum(q, 0)) - (mu - mu_prime) / sigma),
                        Phi((q - (mu ** 2 - 2 * mu * mu_prime) / sigma ** 2) / (2 * mu / sigma)))
