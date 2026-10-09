# Error propagation through a non-linear function: a toy study

This directory studies one question: **how good is linear (first-order) error
propagation?** A quantity $x$ is measured with uncertainty $\sigma$, and we
want the uncertainty on a derived quantity $y = f(x)$. The textbook answer is

$$
\hat y = f(\mu), \qquad \sigma_y = \left|f'(\mu)\right|\,\sigma ,
$$

and in practice the result is quoted as a Gaussian $y \sim \mathcal N\!\left(f(\mu),\,|f'(\mu)|\sigma\right)$.
The scripts check this against the truth with pseudo-experiments (toys) and
with exact calculations, for several functions $f$ and several distributions of $x$.

- `x2_error_propagation.py`: the original study, $y=x^2$ with Gaussian $x$ only.
- `error_propagation_toys.py`: the general study (any $f$, Gaussian, uniform or Poisson $x$).
  One figure per function is written to `plots/<distribution>_<function>.png`.

```
python3 error_propagation_toys.py                         # Gaussian x; x², 1/x, √x, ln x, eˣ
python3 error_propagation_toys.py --dist uniform
python3 error_propagation_toys.py --dist poisson --mu 9
python3 error_propagation_toys.py --funcs x2 sin --mu 1 --sigma 0.8
```

---

## 1. The setup

### The measurement $x$

$x$ always has true mean $\mu$ and standard deviation $\sigma$:

| distribution | definition | $P(\lvert x-\mu\rvert\le\sigma)$ |
|---|---|---|
| Gaussian | $\mathcal N(\mu,\sigma)$ | 68.27% |
| uniform | $\mathcal U[\mu-\sqrt3\sigma,\ \mu+\sqrt3\sigma]$ | $1/\sqrt3 = 57.74\%$ |
| Poisson | $\text{Poisson}(\lambda=\mu)$, so $\sigma=\sqrt\mu$ | depends on $\lambda$ (76.0% for $\lambda=9$) |

The last column already contains one of the main lessons (Section 5).

### The functions $f$

| $f(x)$ | physical example | what makes it interesting |
|---|---|---|
| $x^2$ | energy from a velocity, invariant mass squared | simplest curvature; the bias of the mean is exactly $+\sigma^2$ |
| $1/x$ | momentum from track curvature, time from a rate | pole at $x=0$ |
| $\sqrt x$ | amplitude from a counted rate | concave, defined only for $x\ge 0$ |
| $\ln x$ | log-likelihood, log of a ratio | concave, defined only for $x>0$ |
| $e^x$ | exponential decay or growth | convex, Gaussian $x$ gives a log-normal $y$ |
| $\sin x$, $x^3$ (optional) | angles; odd powers | non-monotonic; inflection point |

### The four things compared in every figure

For each toy measurement $x_i$ the script computes:

1. **$y_i = f(x_i)$:** what nature actually does (blue filled histogram).
2. **$y_{\text{lin},i} = f(\mu) + f'(\mu)(x_i-\mu)$:** the same $x_i$ sent through the
   *tangent line* of $f$ at $\mu$ (red step histogram). This is the hidden
   assumption of linear propagation, made explicit.

It then compares them with:

3. **Linear propagation as a Gaussian,** $\mathcal N(f(\mu), |f'(\mu)|\sigma)$ (red dashed curve).
4. **The exact answer** (solid blue curve and "exact" numbers):
   - the p.d.f. of $y$, from the change of variables over all branches $x_r(y)$ with $f(x_r)=y$:
     $$f_Y(y) = \sum_r f_X\big(x_r(y)\big)\left|\frac{dx_r}{dy}\right| .$$
     For example, $x^2$ gives $[f_X(\sqrt y)+f_X(-\sqrt y)]/(2\sqrt y)$ and $e^x$ gives $f_X(\ln y)/y$.
   - the exact $E[y]$ and $\sigma_y = \sqrt{\operatorname{Var}[y]}$, by integrating over the distribution of $x$.

The toys only illustrate and cross-check: in every case where the exact
quantities exist, toys and exact agree within statistical errors.

---

## 2. Why linear propagation is an approximation, and of what

Linear propagation replaces $f$ by its tangent at $\mu$. Expanding to second order:

$$
f(x) = f(\mu) + f'(\mu)(x-\mu) + \tfrac12 f''(\mu)(x-\mu)^2 + \dots
$$

Taking expectation values:

$$
E[y] \simeq f(\mu) + \tfrac12 f''(\mu)\,\sigma^2 .
$$

This holds for **any** distribution of $x$ with variance $\sigma^2$. Linear propagation drops the
second term, so **the first thing that fails is the central value, not the
error.** The bias is of order $\sigma^2$, while the width correction enters one order later:

$$
\sigma_y^2 \simeq f'^2\sigma^2 + \left(f'f''' + \tfrac12 f''^2\right)\sigma^4
\qquad(\text{Gaussian } x).
$$

The width formula depends on the fourth moment of $x$, so it differs for
uniform or Poisson $x$. For skewed inputs such as Poisson, an additional term $\propto f'f''\mu_3$ appears.

**The linear Gaussian is exact for the wrong variable.** $y_\text{lin}$ is an
affine function of $x$, so for Gaussian $x$ it is *exactly* $\mathcal N(f(\mu),|f'(\mu)|\sigma)$.
The red histogram sits perfectly on the red dashed curve. All the approximation is in
replacing $f$ by its tangent, and none of it is in the Gaussian formula itself.

**The Gaussian shape is an extra assumption.** The propagation formula only gives
two moments, the mean and the variance. For uniform $x$, $y_\text{lin}$ is
**uniform**, not Gaussian (the flat red histogram in `uniform_*.png`). For Poisson $x$
it is **discrete**. The value of $\sigma_y$ is still right to first order, but the shape is not.

---

## 3. Left panel: the distribution of $y$

What to look at:

- **Blue histogram vs blue curve:** toys vs the exact p.d.f. They agree.
- **Blue vs red:** the true distribution of $y$ vs what linear propagation assumes.
  Curvature of $f$ shows up as **skewness**: convex $f$ ($x^2$, $e^x$, $1/x$ for $x>0$)
  gives a long right tail, and concave $f$ ($\sqrt x$, $\ln x$) gives a long left tail.
- **Vertical lines:** $f(\mu)$ (linear prediction, red dotted), exact $E[y]$ (blue dotted),
  and the toy mean (black). The gap between red and blue is the bias.
- **Horizontal bars:** the $\pm1\sigma$ intervals $E[y]\pm\sigma_y$ (blue) and
  $f(\mu)\pm\sigma_\text{lin}$ (red), with their coverage (Section 5).
- **Shaded region at $y<0$:** where the linear Gaussian puts probability on values that
  $y=x^2$ (or $\sqrt x$, $e^x$, ...) can never take. For $x^2$ with $\mu=1$, $\sigma=0.8$,
  that is 27% of the probability.
- **Table:** toy mean and $\sigma$ of $y$ and of $y_\text{lin}$, next to the exact and linear values.
  The printout also gives pulls $(\text{toy}-\text{prediction})/\text{error}$. Pulls of
  order 1 against "exact" and of order 10–100 against "linear" show that the
  disagreement is physics, not a fluctuation.

Example, Gaussian $x$ with $\mu=5$, $\sigma=1$ ($\sigma/\mu = 0.2$):

| $f$ | $f(\mu)$ | exact $E[y]$ | $\sigma_\text{lin}$ | exact $\sigma_y$ |
|---|---|---|---|---|
| $x^2$ | 25 | 26.00 | 10.00 | 10.10 |
| $\sqrt x$ | 2.236 | 2.224 | 0.224 | 0.228 |
| $\ln x$ | 1.609 | 1.588 | 0.200 | 0.212 |
| $e^x$ | 148.4 | 244.7 | 148.4 | 320.8 |
| $1/x$ | 0.200 | does not exist | 0.040 | does not exist |

A 20% relative uncertainty is already enough to bias $e^x$ by 65% of its quoted
error and to underestimate its width by a factor of 2.

---

## 4. Middle panel: when does linear propagation break down?

The input uncertainty $r=\sigma/|\mu|$ is scanned at fixed $\mu$. For Poisson, $\lambda = 1/r^2$
is scanned, since $\sigma/\mu = 1/\sqrt\lambda$. Two quantities are shown, both in units of
the propagated error $\sigma_\text{lin}=|f'(\mu)|\sigma$:

| curve | quantity | meaning | linear propagation predicts |
|---|---|---|---|
| blue | $\sigma_y/\sigma_\text{lin}$ | is the quoted error bar the right size? | 1 |
| orange | $(E[y]-f(\mu))/\sigma_\text{lin}$ | how many error bars is the central value off by? | 0 |

Lines are exact; points are toys. The dotted vertical line marks the setting shown
in the left panel.

**Reading it:** linear propagation is fine where blue stays at 1 **and** orange stays at 0.
The orange curve leaves 0 *linearly* in $r$ while the blue one leaves 1
only *quadratically*, so the bias is what sets the validity range:

| $f$ | bias $/\sigma_\text{lin}$ | $\sigma_y/\sigma_\text{lin}$ |
|---|---|---|
| $x^2$ | $r/2$ (exact) | $\sqrt{1+r^2/2}$ (exact) |
| $\sqrt x$ | $\approx -r/4$ | $\approx 1+\tfrac{7}{16}r^2$ |
| $\ln x$ | $\approx -r/2$ | $\approx 1+\tfrac54 r^2$ |
| $e^x$ | $\approx \tfrac{\mu}{2}\,r$ | explodes, $\propto e^{\sigma^2}$ (log scale on the plot) |
| $1/x$ | $\approx r$ | does not exist for Gaussian $x$ |

(The width column assumes Gaussian $x$.)

**Rule of thumb:** a bias below about 10% of the error needs $|f''|\sigma^2 \lesssim 0.2\,|f'|\sigma$,
that is, $\sigma \lesssim 0.2\,|f'/f''|$. The relevant scale is the
*curvature length* of $f$, not $\mu$.

---

## 5. Right panel: coverage of the $\pm1\sigma$ interval

A "$1\sigma$" error bar is meant to contain the truth 68.27% of the time, but that number
is a property of the Gaussian, not of error bars in general. Two different kinds of
coverage are shown.

### (a) Probability content of a fixed interval

$P\big(y \in [c-s,\ c+s]\big)$ for the true distribution of $y$:

- **blue:** $[E[y]\pm\sigma_y]$, the exact mean and exact $\sigma$;
- **red:** $[f(\mu)\pm\sigma_\text{lin}]$, the linear-propagation interval;
- **grey dash-dot:** the linear interval applied to $y_\text{lin}$. This is just
  $P(|x-\mu|\le\sigma)$ of the *input*, the coverage you would get if $f$ were exactly linear.

### (b) Frequentist (per-experiment) coverage

**green:** each toy is one experiment that measures $x_i$ and quotes
$f(x_i)\pm|f'(x_i)|\sigma$, i.e. linear propagation evaluated at the *measured* value,
since the experimenter does not know $\mu$. The green curve is the fraction of experiments
whose interval contains the true value $f(\mu)$. This is the quantity that matters when an error bar is used for inference.

### The messages

1. **68% is a property of the Gaussian, not of "$1\sigma$".** Even for a perfectly linear $f$,
   the coverage is $P(|x-\mu|\le\sigma)$ of the input: 57.7% for uniform $x$, and 76.0% for Poisson with
   $\lambda=9$, where $k\in[6,12]$. For a discrete $x$ it also jumps as $\lambda$ changes,
   because the interval edges cross integer values.
2. **The right $\sigma$ does not give the right coverage when $y$ is skewed.** For $e^x$
   (log-normal), $E[y]\pm\sigma_y$ contains 91% of the probability, not 68%: the long tail
   inflates $\sigma_y$, while most of the probability sits near the mode. For a skewed $y$, a
   68% interval should be built from quantiles, not from $\pm\sigma$.
3. **Coverage close to 68% does not mean the propagation is right.** The
   red and green curves can cross 68% at values of $r$ where the bias is already large, e.g.
   $e^x$ near $r\approx0.6$. Equally, $1/x$ with Gaussian $x$ has per-toy
   coverage 72% at $r=0.2$, although its mean and variance do not even exist.
4. **Fixed-interval and per-experiment coverage differ.** The per-experiment error bar
   $|f'(x_i)|\sigma$ moves with the measurement: it is wide where $f'$ is large and narrow where $f'$ is small.
   Sharp features in the green curve are real. For $1/x$, the condition
   $|1/x-1/\mu|\le\sigma/x^2$ is quadratic in $x$, and its solution changes from two
   pieces to one at exactly $\sigma/\mu = 1/4$.

Coverages at the default settings (exact values; toys agree within about 0.15%):

| $f$ | Gaussian: $E\pm\sigma_y$ / linear / per-toy | uniform | Poisson($\lambda=9$) |
|---|---|---|---|
| $x^2$ | 68.8 / 69.0 / 68.0 | 58.0 / 59.0 / 57.7 | 76.0 / 68.7 / 66.9 |
| $1/x$ | n/a / 69.2 / 72.0 | 60.9 / 60.1 / 64.6 | 86.2 / 72.0 / 80.3 |
| $\sqrt x$ | 69.0 / 68.2 / 68.7 | 58.0 / 57.7 / 58.4 | 76.0 / 66.9 / 68.7 |
| $\ln x$ | 70.4 / 68.4 / 69.4 | 58.6 / 58.1 / 59.5 | 76.0 / 66.9 / 74.9 |
| $e^x$ | 90.9 / 75.6 / 75.6 | 82.0 / 70.0 / 70.0 | 100 / 70.6 / 67.6 |
| linear $f$ (reference) | 68.27 | 57.74 | 76.0 |

(For Poisson, each experiment uses the true $\sigma=\sqrt\lambda$. A real experimenter
would use $\sqrt{k_i}$, which gives yet another coverage.)

---

## 6. Pitfalls the study exposes

**Moments that do not exist.** For $y=1/x$ with Gaussian $x$, $f_X(0)>0$, so
$E[1/x]=\int f_X(x)/x\,dx$ diverges, and so does the variance. The p.d.f. of $y$ is
perfectly well defined (left panel), but "the mean and $\sigma$ of $y$" are not. The toys
still return finite numbers, because a finite sample never lands close enough to 0, but these
numbers do **not converge** as the number of toys grows. The middle panel shows them jumping
erratically, and the script reports "does not exist". For uniform $x$ the support
is bounded, and the moments exist as long as $0 \notin [\mu-\sqrt3\sigma,\ \mu+\sqrt3\sigma]$.

**Heavy tails.** For $e^x$ with Poisson $x$ ($\lambda=9$),
$\operatorname{Var}[e^x] = e^{\lambda(e^2-1)} - e^{2\lambda(e-1)} \approx 10^{25}$. It is
dominated by values of $k$ so rare that $10^5$ toys essentially never see them, so the
sample $\sigma$ is wrong by orders of magnitude, and so is its own error estimate. The
script flags this. Log-normal $y$ ($e^x$ with Gaussian $x$) shows a milder version: at
large $r$ the toy points in the middle panel fall below the exact curve.

**Domains.** $\sqrt x$ and $\ln x$ are undefined for $x<0$ (or $x\le0$). Measurements there are
discarded and the remaining distribution is renormalised (the "dropped" fraction is
printed). This is itself a modelling choice: a real analysis would have to decide what
an unphysical measurement means.

---

## 7. Take-home messages

1. Linear propagation is a first-order Taylor expansion. Its leading failure is a
   **bias of the central value**, $\tfrac12 f''\sigma^2$, which appears before the error itself goes wrong.
2. The validity condition is $\sigma \ll |f'/f''|$, the curvature scale of $f$, not
   a fixed relative uncertainty.
3. The Gaussian shape of the result is an **additional assumption**. It is exact only for
   Gaussian $x$ *and* linear $f$.
4. "$\pm1\sigma$ = 68%" holds only for Gaussian distributions. For skewed or non-Gaussian
   $y$, quote quantile-based (central 68%) intervals, or propagate the full
   distribution with toys.
5. Before quoting a mean and an uncertainty, check that they **exist** (poles) and that
   the sample estimates have **converged** (heavy tails).
