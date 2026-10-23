# Confidence intervals and exclusion limits in HEP: a toy study

This directory explains, and lets you simulate with toy datasets, **what a
confidence interval / confidence limit is** and **how exclusion limits are set
in high-energy physics** (CL$`_{s+b}`$, CLs, the profile-likelihood statistic
$`\tilde q_\mu`$, asymptotic formulae, expected limits and the "Brazil" bands).
Every statement is checked with pseudo-experiments, and every number below was
printed by the scripts.

```
python3 01_neyman_coverage.py              # Neyman belts, coverage, 2D regions
python3 02_boundaries_feldman_cousins.py   # physical boundaries, flip-flop, Feldman-Cousins, Bayes
python3 03_cls_counting.py                 # CL_s+b vs CLs, expected limits, counting with a systematic
python3 04_profile_likelihood_limits.py    # q~_mu, toys vs asymptotics, Brazil plot vs mass
python3 05_discovery_q0.py                 # q0, Z_A vs s/sqrt(b), look-elsewhere effect
jupyter lab confidence_limits_walkthrough.ipynb   # the same story, condensed, with plots inline
```

Needs `numpy`, `scipy`, `matplotlib` (no ROOT). Each script takes `--ntoys`,
`--seed`, `--outdir` (and more, see `--help`) and runs in 5–30 s. The library is
`cltoys.py`; it is vectorised over toys and is meant to be reused:

```python
import numpy as np, cltoys as ct
model, edges = ct.bump_model(mass=125, s_tot=20, b_tot=1000, nuisance=("lnN", 1.10))
n, a = model.generate(mu=0.0, theta=0.0, ntoys=5000, rng=np.random.default_rng(1))   # b-only toys
ts = model.test_statistics(1.0, n, a)                   # t_mu, q_mu, qtilde_mu for every toy
ct.asymptotic_expected_limit(model, cl=0.95, n_sigma=0) # median expected CLs limit (Asimov)
```

---

## 0. Where to look in STAMET (from the knowledge index)

| Topic | Primary source | Also |
|---|---|---|
| What a confidence interval is; Neyman belt; coverage | **book `confidenceIntervals.ipynb` c1–c2**; 2024/Lecture_9-2 p.3–10 | Cowan §9.1–9.2 (PDF p.131–132); Terascale 06 Cowan p.3–5 |
| Gaussian and Poisson intervals, $`n_{obs}=0`$ → 3 events, empty-bin error bar 1.8 | book c2; Lecture_9-2 p.11–17 | Cowan §9.3–9.4, §9.9 (p.136–139, 152); interactive nbs `Poisson_CI`, `Gaussian_CI` |
| Intervals from $`\Delta\ln L`$ / $`\Delta\chi^2`$; 2D regions (39.3% vs 68.3%) | book c3; Lecture_9-2 p.20–22 | Cowan §9.6–9.7 (p.143–145) |
| Limits near a physical boundary; Bayesian upper limits | book c4, c26; Lecture_13-2 p.5–18 | Cowan §9.8 (p.149); interactive `LimitNearBoundariesBayesianApproach(+Poisson)` |
| Flip-flop, Feldman–Cousins (Poisson + b, Gaussian at 0), sensitivity | book c5–c11; Lecture_13-2 p.19–57 | **papers/Feldman-Cousins** §III–VI (p.3–16, tables p.7–11, Table XII p.16); interactive `FC_PoissonMeanWithKnownBackground` |
| LEP test statistic $`Q=L_{s+b}/L_b`$, $`-2\ln Q`$; **CLs** | book c12, **c14**; Lecture_9-2 p.24–46 | papers/higgs_statistics App. A (p.34–40); Gross p.7–8, 12–13; C:what_combine_does/statistical_tests L173 |
| Profile likelihood ratio, nuisances, profiling vs marginalising | book c13, c15–c16; May14 p.2–12 | 1007.1727 §2 (p.2–8); C:fitting_concepts L42–57; Combine paper p.4–5 |
| $`q_\mu`$, $`\tilde q_\mu`$, expected limits, ±1σ/±2σ bands | **book c20**; **May14 p.34–43** | 1007.1727 §2.4–2.5, §3.6–3.7, **§4.3 (p.21)**; Terascale 06 p.30 (reading Brazil plots); C:commonstatsmethods L7, L93 |
| Wald/Wilks, asymptotic formulae, Asimov data set | book c18–c19; May14 p.21–32 | **1007.1727 §3 (p.8–17)**; Cowan Weizmann L3 p.13–26 |
| Discovery $`q_0`$, $`Z_A`$ vs $`S/\sqrt B`$ | book c17, c22; May14 p.13–25, 46–47 | 1007.1727 §5.1 (p.22–26) |
| Look-elsewhere effect | book c27; May14 p.52–58 | C:part3/nonstandard L1116; Cowan Weizmann L3 p.45–52 |
| CMS practice (Combine): toys, HybridNew, AsymptoticLimits | C:part3/commonstatsmethods L7, L297, L522; C:part3/runningthetool L141–215 | Combine paper §6.3–6.4 (p.24–30); **`data/benchmarks/simple-counting/lhcClsByHand.cxx`**; C:part5/longexerciseanswers |
| Real analyses | Iemmi thesis p.143–171 (q₀, q_μ, asymptotics, CLs, blinded limits) | Steggemann p.157–185; Rieger §8.2 (p.144–157); Berger p.28–31 |

(C: = `HiggsAnalysis-CombinedLimit/docs/…` line; book = `hep-datanalysis-jb/book/`; May14 = `2024/STAMET_May14_2024.pdf`.)

---

## 1. What a confidence interval is

You measure data $`x`$ (or an estimator $`\hat\theta`$) whose p.d.f. $`f(x\mid\theta)`$
depends on an unknown parameter $`\theta`$. A **confidence interval** at
confidence level (CL) $`1-\alpha`$ is a *rule* $`x \mapsto [a(x), b(x)]`$ such that

```math
P\big(a(x) \le \theta \le b(x) \,\big|\, \theta\big) \;\ge\; 1-\alpha \qquad \text{for every } \theta .
```

The random object is the interval, not $`\theta`$. "95% CL" is a property of the
*procedure*: if the experiment were repeated, 95% of the intervals would contain
the true value. This frequency is the **coverage**; a procedure **over-covers**
(conservative) or **under-covers** if its coverage is above or below $`1-\alpha`$
for some $`\theta`$. A statement about one interval, "θ is in [a, b] with 95%
probability", is a Bayesian credible interval, a different object.

**Neyman construction** (book c2, Lecture_9 p.5–9). For each possible true
$`\theta`$ choose an *acceptance region* $`A(\theta)`$ of the observable with
$`P(x\in A(\theta)\mid\theta) = 1-\alpha`$ ("build horizontally"). The
**confidence belt** is the union of these regions. For the observed $`x_{obs}`$ the
interval is $`\{\theta : x_{obs}\in A(\theta)\}`$ ("read vertically"). Coverage
holds by construction: $`\theta_{true}`$ is in the interval exactly when $`x`$ fell
in $`A(\theta_{true})`$, which happens with probability $`1-\alpha`$.

The construction leaves one choice open: **which** values of $`x`$ go into
$`A(\theta)`$. This is the **ordering principle**:

- central interval: leave $`\alpha/2`$ in each tail;
- upper limit: reject only *small* $`x`$ (a deficit relative to $`\theta`$), so $`\theta_{up}`$ is the value for which $`P(x \le x_{obs}\mid\theta_{up}) = \alpha`$;
- Feldman–Cousins: rank $`x`$ by the likelihood ratio (Section 3).

**Gaussian estimator**, $`\hat\theta\sim N(\theta,\sigma)`$: the belt is
$`\theta-z\sigma\le\hat\theta\le\theta+z\sigma`$, giving
$`\hat\theta\pm z\sigma`$ with $`z=\Phi^{-1}(1-\alpha/2)`$; the one-sided upper limit is
$`\theta_{up}=\hat\theta+\sigma\Phi^{-1}(1-\alpha)`$.

**Poisson count** $`n\sim\text{Pois}(\lambda)`$ (exact, "Garwood"):

```math
\sum_{k\ge n_{obs}} \text{Pois}(k;\lambda_1)=\tfrac{\alpha}{2},\quad
\sum_{k\le n_{obs}} \text{Pois}(k;\lambda_2)=\tfrac{\alpha}{2}
\;\;\Longleftrightarrow\;\;
\lambda_1=\tfrac12\chi^2_{\alpha/2}(2n_{obs}),\;\; \lambda_2=\tfrac12\chi^2_{1-\alpha/2}(2n_{obs}+2).
```

With $`n_{obs}=0`$ the upper limit solves $`e^{-\lambda_{up}}=\alpha`$, so
$`\lambda_{up}=-\ln 0.05 = 2.996`$ at 95% CL: "zero events → limit of 3 events".

### What the toys show (`01_neyman_coverage.py`)

| check | result |
|---|---|
| Poisson $`n=9`$, 90% central | $`[4.695,\,15.705] = 9^{+6.7}_{-4.3}`$ (book c2 figure) |
| $`n=0`$ upper limit at 68.27 / 90 / 95% CL | 1.148 / 2.303 / 2.996 |
| exact 68.27% Poisson interval, coverage over $`\lambda\in(0,15]`$ | between 0.689 and 0.99: **always ≥ 68.27%**, saw-tooth because $`n`$ is discrete |
| naive $`n\pm\sqrt n`$ | coverage down to 0.05 at small $`\lambda`$ (30% at $`\lambda=0.5`$): **under-covers** |
| toys (20 000 per point) vs exact sums | agree within the binomial error ($`\pm0.003`$) |
| 2D Gaussian, $`\rho = 0.6`$: $`P(\Delta\chi^2\le1)`$ | 0.391 (exact 0.393) — **not** 68.3%; $`\Delta\chi^2=2.30`$ gives 0.683 |

Lesson: discreteness forces over-coverage (you cannot hit 68.27% exactly); a
recipe that looks reasonable ($`\pm\sqrt n`$) can badly under-cover; in $`k`$
dimensions the $`\Delta\chi^2`$ threshold is the $`\chi^2_k`$ quantile (2.30, 6.18 for
68.3%, 95.4% with $`k=2`$).

Plots: `plots/01a_neyman_belts.png`, `01b_poisson_coverage.png`, `01c_2d_regions.png`.

---

## 2. HEP searches: intervals ↔ tests, limits ↔ exclusion

In a search the parameter is a **signal strength** $`\mu`$ (signal cross section
in units of a reference, e.g. $`\sigma/\sigma_{SM}`$) with expected yields

```math
E[n_i] = \mu\, s_i + b_i(\boldsymbol\theta),
```

and the physics constrains $`\mu\ge0`$. Every confidence interval is the set of
parameter values **not rejected** by a test of size $`\alpha`$ ("inverting a test",
Terascale 06 p.3–4). Two questions are asked of the same data:

- **Discovery**: test $`H_0:\mu=0`$. Reject if $`p_0 < 2.87\times10^{-7}`$ (5σ).
- **Exclusion (upper limit)**: for each $`\mu`$ test $`H_\mu`$ against the
  alternative "less signal". The **upper limit** $`\mu_{up}`$ is the
  smallest $`\mu`$ rejected at $`\alpha=0.05`$; all $`\mu>\mu_{up}`$ are
  **excluded at 95% CL**. An **exclusion limit** is just this one-sided upper
  confidence limit, read as "signals this large would have produced a larger
  excess than we saw, in more than 95% of the experiments".

A model is "excluded" when its prediction is above the limit (e.g. $`\mu=1`$
excluded if $`\mu_{up}<1`$). Scanned vs a model parameter (mass, $`\kappa_\lambda`$, …)
this gives the familiar exclusion plots.

---

## 3. Limits near a physical boundary (`02_boundaries_feldman_cousins.py`)

With $`\mu\ge0`$ and a background-like downward fluctuation, the classical
upper limit can be **negative** or the interval **empty**:

- Gaussian, $`x_{obs}=-2\sigma`$: $`\mu_{up}^{90\%} = x+1.28 = -0.72`$.
- Poisson with $`b=3`$, $`n_{obs}=0`$: the 90% classical UL is $`-0.70`$, the
  central interval is empty (book c7, Lecture_13 p.28–34).

Both are correct frequentist statements (they happen in 5–10% of experiments),
but useless as physics. Three remedies:

**Flip-flop** (FC paper §III, book c6): "quote an upper limit if $`x<3\sigma`$,
a central interval otherwise, and use $`\max(x,0)`$". Deciding the type of
interval after seeing the data breaks the construction. The script finds
coverage **85.0%** for $`1.3<\mu<4.6`$ (toys at $`\mu=2`$: $`0.849\pm0.002`$), the
FC paper's 85% for $`1.36<\mu<4.28`$.

**Feldman–Cousins** (FC paper §IV, book c7–c8): Neyman construction with the
ordering

```math
R(x) = \frac{L(x\mid\mu)}{L(x\mid\mu_{best})},\qquad \mu_{best}=\max(0,\hat\mu),
```

i.e. accept the $`x`$ values that are most likely *relative to the best
physically allowed fit*. The intervals change smoothly from upper limits to
two-sided ones (no flip-flop), are never empty and cover by construction.

| check (90% CL) | script | reference |
|---|---|---|
| Gaussian: $`x=-2,-1,0,1,2,3`$ | [0, 0.40], [0, 0.81], [0, 1.64], [0, 2.64], [0.59, 3.64], [1.37, 4.64] | FC Table X: identical except [0.58, 3.64] at $`x=2`$ (0.01 grid) |
| Poisson b = 3, $`n=1\ldots10`$ | [0,1.88], [0,3.04], [0,4.42], [0,5.59], [0,6.99], [0.15,8.46], [0.90,9.53], [1.51,10.99], [1.88,12.29], [2.64,13.50] | FC Table IV: identical (±0.01 grid) |
| Poisson b = 3, $`n=0`$ | [0, 0.95] raw; **[0, 1.075]** with `--fc-monotone` | FC Table IV: [0, 1.08] |
| acceptance region for $`\mu=0.5`$, b = 3 | $`n\in[0,6]`$, P = 0.935 | book c7: 0.935 |
| FC Poisson exact coverage, $`\mu\in[0,10]`$ | min 0.9000 | ≥ 90% |

The $`n=0`$ entry needs FC's second fix (paper p.6): they force $`\mu_{up}(n,b)`$ to
be non-increasing in $`b`$, which lengthens it from 0.95 to 1.08. The book (c7)
does not mention this.

**Bayesian** (book c4, c26): flat prior $`\pi(\mu)=1`$ for $`\mu\ge0`$; then
$`\int_0^{\mu_{up}} p(\mu\mid x)\,d\mu = 1-\alpha`$. For the Gaussian this gives
$`\mu_{up}=x+\Phi^{-1}\!\big(1-\alpha\,\Phi(x)\big)`$, which is **exactly the CLs
limit** (Section 4).

**Sensitivity.** A worse experiment with a lucky deficit quotes a better limit:
FC gives 2.44 for ($`b=0,n=0`$) but 1.83 for ($`b=10,n=5`$) (book c11). FC
therefore recommend quoting the **sensitivity** = mean upper limit of
background-only experiments. Toys give 2.43 / 3.27 / 4.41 / 5.17 / 6.73 for
$`b`$ = 0 / 1 / 3 / 5 / 10 (FC Table XII: 2.44 / 3.28 / 4.42 / 5.18 / 6.76). The
LHC answer to the same problem is the expected limit with its bands (Section 6).

Plots: `plots/02a_gaussian_boundary.png`, `02b_poisson_fc.png`.

---

## 4. CL$`_{s+b}`$ and CLs (`03_cls_counting.py`)

Take a test statistic $`Q`$ ordered so that small values look background-like.
For a single-bin count, $`n`$ itself (equivalent to the LEP $`-2\ln Q`$,
$`Q=L_{s+b}/L_b`$, book c12). Define (book c14, Lecture_9 p.39–46)

```math
CL_{s+b} = P(Q\le Q_{obs}\mid s+b) = p_{s+b},\qquad
CL_b = P(Q\le Q_{obs}\mid b) = 1-p_b,\qquad
CL_s = \frac{CL_{s+b}}{CL_b}.
```

- **Classical (CL$`_{s+b}`$) exclusion**: exclude $`s`$ if $`CL_{s+b}<\alpha`$. This is
  the Neyman upper limit with exact coverage, but with no sensitivity
  ($`s\to0`$) it still excludes in 5% of background-only experiments.
  Excluding a signal you cannot distinguish from no signal is a **spurious
  exclusion**.
- **CLs**: exclude $`s`$ if $`CL_s<\alpha`$. Dividing by $`CL_b`$ penalises
  downward fluctuations of the background. When the hypotheses separate well,
  $`CL_b\to1`$ and $`CL_s\to CL_{s+b}`$; when they do not, $`CL_s\to1`$ and nothing
  is excluded. CLs is a ratio of p-values, not a p-value, and **over-covers**.

What the script prints ($`b=3`$, 95% CL):

| $`n_{obs}`$ | CL$`_{s+b}`$ UL | CLs UL | Bayes, flat prior |
|---|---|---|---|
| 0 | −0.004 | 2.996 | 2.996 |
| 1 | 1.744 | 3.643 | 3.643 |
| 2 | 3.296 | 4.443 | 4.443 |
| 5 | 7.513 | 7.663 | 7.663 |
| 10 | 13.962 | 13.963 | 13.963 |

| P(exclude $`s`$ ∣ background only), $`b=3`$ | $`s=0.01`$ | $`s=1`$ | $`s=4`$ | $`s=6`$ |
|---|---|---|---|---|
| CL$`_{s+b}`$ | 0.050 | 0.050 | 0.42 | 0.65 |
| CLs | 0.000 | 0.000 | 0.20 | 0.65 |

Coverage of the CLs limit: 1.000 at $`s=0`$, 0.9975 at $`s=3`$, 0.974 at $`s=10`$
(nominal 0.95).

Two facts worth remembering:

1. For a Poisson count with known $`b`$, **CLs ≡ Bayesian upper limit with a flat
   prior** on $`s\ge0`$, for every $`n`$ and $`b`$ (because
   $`P(n\le n_{obs}\mid\lambda)`$ is the upper incomplete gamma function). With
   $`n_{obs}=0`$ both give 2.996 whatever $`b`$ is. The book (c4) calls the
   agreement for $`n_{bkg}=0`$ "a pure coincidence"; it is an identity.
2. For a Gaussian estimator with boundary, the CLs UL is
   $`x+\Phi^{-1}(1-\alpha\Phi(x))`$, also identical to the flat-prior Bayes UL.

Plots: `plots/03a_cls_counting.png`, `03b_expected_limits_counting.png`, `03c_cls_by_hand.png`.

---

## 5. The LHC recipe: profile likelihood, nuisances, toys and asymptotics

**Likelihood** (book c16, Combine paper eq. 1–2): with nuisances $`\boldsymbol\theta`$
constrained by auxiliary measurements ("global observables" $`a`$),

```math
L(\mu,\theta) = \prod_i \text{Pois}\big(n_i;\ \mu s_i + b_i\,\kappa^{\theta}\big)\ \cdot\ \mathcal N(a;\theta,1)
```

(here one log-normal normalisation nuisance, Combine's `lnN`).

**Profile likelihood ratio** $`\lambda(\mu)=L(\mu,\hat{\hat\theta}(\mu))/L(\hat\mu,\hat\theta)`$.
Nuisances are fitted (profiled) at each tested $`\mu`$. The LHC upper-limit
statistic (C:statistical_tests L173; 1007.1727 §2.5) is

```math
\tilde q_\mu=\begin{cases}
-2\ln\dfrac{L(\mu,\hat{\hat\theta}(\mu))}{L(0,\hat{\hat\theta}(0))} & \hat\mu<0\\[2ex]
-2\ln\dfrac{L(\mu,\hat{\hat\theta}(\mu))}{L(\hat\mu,\hat\theta)} & 0\le\hat\mu\le\mu\\[2ex]
0 & \hat\mu>\mu
\end{cases}
```

$`\hat\mu>\mu`$ gives 0 because an *excess* is never evidence against $`\mu`$
(one-sided); $`\hat\mu<0`$ is replaced by the boundary to avoid the
negative-limit pathology of Section 3.

**p-values from toys** (Combine `HybridNew --LHCmode LHC-limits`): fit the data
to get $`\hat{\hat\theta}(\mu)`$ and $`\hat{\hat\theta}(0)`$; generate s+b toys at
$`(\mu,\hat{\hat\theta}(\mu))`$ and b-only toys at $`(0,\hat{\hat\theta}(0))`$, with the
global observable $`a`$ drawn around those values ("frequentist toys",
C:runningthetool L191); recompute $`\tilde q_\mu`$ for each toy, then

```math
p_\mu = P(\tilde q_\mu \ge \tilde q_{\mu,obs}\mid\mu),\quad
1-p_b = P(\tilde q_\mu \ge \tilde q_{\mu,obs}\mid 0),\quad
CL_s(\mu) = \frac{p_\mu}{1-p_b},
```

and scan $`\mu`$ until $`CL_s=0.05`$.

**Asymptotics** (Wald, 1007.1727 §3): for large samples
$`-2\ln\lambda(\mu)\approx(\mu-\hat\mu)^2/\sigma^2`$ with $`\hat\mu\sim N(\mu',\sigma)`$, so
$`f(\tilde q_\mu\mid\mu')`$ is known in closed form. $`\sigma`$ comes from the
**Asimov data set** (data = expectation under $`\mu'=0`$):
$`\sigma_A^2=\mu^2/\tilde q_{\mu,A}`$. Then

```math
p_\mu = 1-\Phi\big(\sqrt{\tilde q_\mu}\big),\quad 1-p_b = 1-\Phi\big(\sqrt{\tilde q_\mu}-\mu/\sigma\big)\qquad (\tilde q_\mu\le\mu^2/\sigma^2),
```

```math
p_\mu = 1-\Phi\!\Big(\frac{\tilde q_\mu+\mu^2/\sigma^2}{2\mu/\sigma}\Big),\quad
1-p_b = 1-\Phi\!\Big(\frac{\tilde q_\mu-\mu^2/\sigma^2}{2\mu/\sigma}\Big)\qquad (\tilde q_\mu>\mu^2/\sigma^2).
```

This is Combine's `AsymptoticLimits`; `cltoys.asymptotic_cls` implements it.

### What the toys show

Bump hunt (`04_profile_likelihood_limits.py`): 1000 background events in
[100, 180] GeV, Gaussian signal (σ = 2 GeV, 20 events for $`\mu=1`$), 10% lnN on
the background; pseudo-data with $`\mu=1.5`$ injected at 150 GeV.

| $`m=125`$ GeV, $`\mu=1.18`$ | $`p_\mu`$ | $`1-p_b`$ | CLs |
|---|---|---|---|
| toys (4000 + 4000) | 0.1345 | 0.795 | 0.169 ± 0.007 |
| asymptotic | 0.1208 | 0.785 | 0.154 |

| 95% CL limit on $`\mu`$, $`m=125`$ GeV | −2σ | −1σ | median | +1σ | +2σ | observed |
|---|---|---|---|---|---|---|
| toys ("expected from grid") | 0.708 | 0.925 | 1.175 | 1.697 | 2.253 | 1.585 |
| asymptotic | 0.616 | 0.836 | 1.180 | 1.681 | 2.317 | 1.561 |

With about 20 signal events on about 50 background events under the peak, the
asymptotic formulae are already good to a few %, except in the low tail of the
expected band, which the toy grid resolves poorly.

At **low counts they fail** (`03_cls_counting.py`, counting experiment, b = 3):

| expected 95% CLs UL | 2.5% | 16% | 50% | 84% | 97.5% |
|---|---|---|---|---|---|
| toys | 2.996 | 3.643 | 5.395 | 7.663 | 10.17 |
| asymptotic | 2.208 | 3.151 | 4.779 | 7.468 | 11.39 |
| toys, b = 100 | 11.60 | 15.29 | 21.37 | 29.77 | 39.78 |
| asymptotic, b = 100 | 10.89 | 14.79 | 20.90 | 29.81 | 41.15 |

The −2σ toy band at b = 3 sits at 2.996: no limit can go below the $`n=0`$ value.

**Combine's "CLs by hand" benchmark** (`lhcClsByHand.cxx`: $`N_{obs}=4`$,
$`B=3.26`$, κ = 1.2353): $`\hat r=0.74`$; at $`r=1`$ toys give CLs = 0.723 ± 0.009,
asymptotic 0.705; 95% CL limit $`r<6.00`$ (toys), 5.82 (asymptotic).
Surprise: the stat-only CLs limit is 6.31, *higher* than with the systematic.
The cause is discreteness. A continuous global observable breaks the ties among
toys with $`n=n_{obs}`$, so the toy p-values become ≈ $`P(n<n_{obs})+\tfrac12P(n=n_{obs})`$
("mid-p") instead of $`P(n\le n_{obs})`$. With κ = 1.001 the toys already give
CLs(r = 6) = 0.049 vs 0.061 stat-only. The stat-only discrete limit is the
conservative one.

---

## 6. Expected limits and the Brazil plot

The **expected limit** is the median upper limit you would get if there were
no signal. The **bands** are the 16–84% (±1σ, green) and 2.5–97.5% (±2σ,
yellow) quantiles of the distribution of upper limits under background-only
toys (May14 p.39; 1007.1727 §4.3; the book's c20 says 5/95%, see the caveats
below). Three ways to get them, all implemented:

1. **Brute force**: generate b-only toys, compute each one's limit, take
   quantiles (`03_cls_counting.py (d)`, `cltoys.per_toy_asymptotic_limits`).
2. **From the grid** (Combine `HybridNew --expectedFromGrid k`): at each tested
   $`\mu`$ replace $`\tilde q_{obs}`$ by the $`(1-k)`$ quantile of $`f(\tilde q_\mu\mid0)`$,
   because a larger $`\tilde q`$ means a lower limit (`cltoys.toy_limit_scan`).
3. **Asymptotic** (1007.1727 eq. 89): $`\mu_{up}^{\pm N}=\sigma_A\,[\Phi^{-1}(1-\alpha\Phi(N))\pm N]`$,
   median $`=1.96\,\sigma_A`$ for CLs. Combine `AsymptoticLimits` (`--run blind`
   uses the pre-fit Asimov data).

**Reading the Brazil plot** (`plots/04d_brazil_plot.png`): the dashed line is
the expected limit, the bands its spread, and the solid line the observed limit.
At 150 GeV the observed limit (2.09) is above the +2σ band (1.64), matching the
local excess $`Z=\sqrt{q_0}=3.1`$ in the lower panel. Where the observed curve
dips below the band, the background fluctuated low. An observed limit below 1 at a
given mass excludes the $`\mu=1`$ model there. Toy limits at 125 and 150 GeV
(1.54, 2.07) agree with the asymptotic curve (1.56, 2.09).

---

## 7. Discovery and the look-elsewhere effect (`05_discovery_q0.py`)

- $`q_0=-2\ln\lambda(0)`$ for $`\hat\mu\ge0`$, else 0. Under background only,
  $`f(q_0\mid0)=\tfrac12\delta(q_0)+\tfrac12\chi^2_1`$ and $`Z=\sqrt{q_0}`$. Toys:
  50.6% at zero; $`P(Z>1,2,3)`$ = 0.155, 0.0214, 0.0017 vs 0.159, 0.0228, 0.00135.
- Median discovery significance of a counting experiment:
  $`Z_A=\sqrt{2((s+b)\ln(1+s/b)-s)}`$. It equals $`s/\sqrt b`$ only for $`s\ll b`$:
  for b = 1, s = 5, $`s/\sqrt b=5`$ but $`Z_A`$ = 3.39 (toy median 3.39).
- LEE: the observed excess of the bump hunt has local $`Z=3.25`$
  ($`p=5.7\times10^{-4}`$). Background-only toys scanned over the same 29 masses give
  $`p_{global}=0.0135\pm0.003`$ ($`Z_{global}=2.2`$, trial factor ≈ 24).

---

## 8. Mapping to Combine

| here | Combine |
|---|---|
| `asymptotic_observed_limit`, `asymptotic_expected_limit` | `combine -M AsymptoticLimits card.txt` (`--run blind`: pre-fit Asimov) |
| `toy_cls_point`, `toy_limit_scan` | `combine -M HybridNew --LHCmode LHC-limits card.txt`, which means `--testStat=LHC --rule CLs --generateNuisances=0 --generateExternalMeasurements=1 --fitNuisances=1` (`src/HybridNew.cc` L144): exactly the frequentist toys of Section 5. Expected limits: `--expectedFromGrid 0.5` (0.16, 0.84, 0.025, 0.975 for the bands; C:commonstatsmethods L517). Without `--LHCmode` the default test statistic is LEP (index §8.12) |
| `model.asimov(0)` | `-t -1` (with `--expectSignal 0`) |
| `q0`, `05_discovery_q0.py` | `-M Significance` (asymptotic) or `HybridNew --LHCmode LHC-significance` |
| FC intervals | `-M HybridNew --LHCmode LHC-feldman-cousins`; `test/makeFCcontour.py` |

---

## 9. Problems found in the sources while preparing this

1. **book `confidenceIntervals` c2**: "P(v_β(θ) ≤ θ̂ ≤ u_α(θ)) **<** 1−α−β" should
   be "= 1−α−β" (≥ for a discrete observable).
2. **book c7**: the classical UL is written as $`0.1=\sum_0^{n_{obs}-1}P(n\mid\mu+b)`$;
   it must be $`\sum_0^{n_{obs}}`$ (c2 has it right). The central-interval
   equations in the same cell are also garbled. In the FC worked example
   "μ = 0.05" is a typo for μ = 0.5 (the probabilities are those of μ = 0.5).
3. **book c7 / FC table**: the raw FC construction gives [0, 0.95] for b = 3,
   n = 0; the published 1.08 includes FC's monotone-in-b correction (paper p.6).
4. **book c20**: "median ± 2σ (5%, 95% quantiles)" should be 2.5% and 97.5%
   (as on May14 p.39 and in 1007.1727).
5. **book c4**: the Bayesian flat-prior and classical limits agreeing for
   $`n_{bkg}=0`$ is not "a pure coincidence". For a Poisson count the flat-prior
   limit equals the CLs limit for any $`b`$, and CLs = CL$`_{s+b}`$ when $`b=0`$.
   The Poisson formula in that cell also misses the minus sign in $`e^{-(\nu_s+\nu_b)}`$.
6. **Combine `lhcClsByHand.cxx`** does not compile as written: the last lines
   use `sqrt(*(1.-OnemPb)…)` and `OnemPg` (should be `OnemPb*(1.-OnemPb)` and
   `OnemPb`). Its fit also bounds $`r\in[0,r_0]`$, i.e. it uses $`\tilde q_\mu`$.

---

## Files

- `cltoys.py`: library (Neyman/Poisson/Gaussian intervals, FC, counting CLs,
  `BinnedModel` with vectorised fits and LHC test statistics, toy and
  asymptotic CLs, expected limits, `bump_model`).
- `01_…05_*.py`: the studies above; plots in `plots/`, printouts in `logs/`.
- `confidence_limits_walkthrough.ipynb`: condensed walkthrough, executed.
