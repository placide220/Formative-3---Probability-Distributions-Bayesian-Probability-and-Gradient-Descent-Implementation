# Probability, Bayesian Inference & Gradient Descent — Assignment Package

**Datasets used:** `GaltonFamilies.csv` (Part 1), `IMDB_Dataset.csv` — 50,000 reviews (Part 2).

---

## 📁 Files in this package

| File | Part | Description |
|---|---|---|
| `em_heights.py` | 1 | EM algorithm from scratch on real Galton heights (Father vs Children) |
| `part1_means_over_iterations.png` | 1 | Plot: μ1, μ2 convergence over EM iterations |
| `part1_loglikelihood_over_iterations.png` | 1 | Plot: log-likelihood convergence |
| `part2_bayesian_sentiment.py` | 2 | Bayes' Theorem from scratch on real IMDb reviews |
| `part3_manual_gradient_descent.pdf` | 3 | Neat, typed manual calculations — full ML pipeline, 4 iterations, b as vector |
| `part4_gradient_descent_code.py` | 4 | Gradient descent in code, SciPy-verified gradients, same ML pipeline structure |
| `part4_parameters_over_iterations.png` | 4 | Plot: m1, m2, b1, b2 over iterations |
| `part4_error_over_iterations.png` | 4 | Plot: MSE cost over iterations |

**For Colab:** upload `GaltonFamilies.csv` and `IMDB_Dataset.csv` to your session (or mount Drive) —
both scripts default to relative filenames and will find them automatically.

---

## Part 1 — EM Algorithm (Fathers vs Children Heights)

**Dataset:** 935 real Galton family records — 205 unique fathers + 934 children, pooled into
1,139 unlabeled heights (in inches). EM is never told which value came from which group.

### Should we just split at the global mean?

**No.** Father and child heights overlap substantially — a tall child can easily be taller than a
short father. A hard threshold at the global mean (67.21") misclassifies every point in that overlap
zone and biases both sub-group means. EM instead assigns every point a **soft probability** of
membership in each group and refines its estimate iteratively, which is provably more accurate on
overlapping distributions.

**Concrete comparison from this run:**

| Method | Group 1 mean | Group 2 mean |
|---|---|---|
| Naive mean-split | 64.26" (n=575) | 70.21" (n=564) |
| EM (converged) | **64.7354"** (μ1) | **69.6765"** (μ2) |

### Optimization Tracking Table (real output)

| Iteration | μ1 (Children) | μ2 (Fathers) | σ1² | σ2² | π1 | π2 | Log-Likelihood |
|---|---|---|---|---|---|---|---|
| 0 (init) | 64.5000 | 70.0000 | 12.7241 | 12.7241 | 0.5000 | 0.5000 | -3119.8222 |
| 1 | 65.2205 | 69.2054 | 9.0603 | 8.4468 | 0.5011 | 0.4989 | -3057.9135 |
| 2 | 65.1609 | 69.2600 | 8.9420 | 8.1040 | 0.5005 | 0.4995 | -3056.9719 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 10 (converged) | 64.7354 | 69.6765 | 7.0185 | 6.2236 | 0.4995 | 0.5005 | -3049.2367 |

Log-likelihood rises sharply after initialization and flattens out — this is the convergence signal
we stop on.

### Live Classification Demo
```
Test height: 68 inches
  P(Child)  = 0.3553
  P(Father) = 0.6447
```
(Replace `test_height` in the script with whatever height the coach gives you live.)

### Convergence Plots

![Means over iterations](part1_means_over_iterations.png)

![Log-likelihood convergence](part1_loglikelihood_over_iterations.png)

---

## Part 2 — Bayesian Probability (IMDb Sentiment)

**Dataset:** real IMDb Movie Reviews — 50,000 reviews (25,000 positive / 25,000 negative).
**Computed:** P(Positive | keyword) only, per assignment instructions.
**Prior:** P(Positive) = 0.5000 for every keyword (balanced dataset).

### Keywords chosen as POSITIVE indicators

| Keyword | P(Positive) | P(kw\|Pos) | P(kw) | P(Pos\|kw) | n(kw) |
|---|---|---|---|---|---|
| excellent | 0.5000 | 0.1147 | 0.0710 | **0.8074** | 3552 |
| brilliant | 0.5000 | 0.0635 | 0.0418 | **0.7601** | 2088 |
| masterpiece | 0.5000 | 0.0351 | 0.0241 | **0.7274** | 1207 |
| love | 0.5000 | 0.2266 | 0.1785 | **0.6349** | 8924 |

### Keywords chosen as NEGATIVE indicators

| Keyword | P(Positive) | P(kw\|Pos) | P(kw) | P(Pos\|kw) | n(kw) |
|---|---|---|---|---|---|
| boring | 0.5000 | 0.0236 | 0.0610 | **0.1937** | 3051 |
| waste | 0.5000 | 0.0070 | 0.0507 | **0.0691** | 2534 |
| awful | 0.5000 | 0.0114 | 0.0577 | **0.0985** | 2883 |
| worst | 0.5000 | 0.0164 | 0.0887 | **0.0927** | 4434 |

**Interpretation:** every positive-indicator keyword pulls the posterior well above the 0.50 prior
(up to 0.81 for "excellent"), while every negative-indicator keyword pulls it well below (down to
0.069 for "waste") — exactly the contrast that validates these as strong sentiment signals. "love"
has the weakest lift of the positive set despite the largest sample size (n=8924), since it's used
more loosely/generically in reviews than words like "masterpiece."

**Keyword justification:** "excellent," "brilliant," and "masterpiece" are strong, unambiguous
praise words rarely used sarcastically; "love" was included as a high-frequency emotional signal.
"boring," "waste," "awful," and "worst" are direct, common complaint words with little ambiguity.

---

## Part 3 — Manual Gradient Descent (see PDF for full derivation)

**Model:** ŷ = m₁x₁ + m₂x₂ + b, where **b = [b₁, b₂] is a 2-element vector** — one independent bias
per data point (per the original problem statement's `Initial b = [1,1]`), NOT a shared scalar.

Every iteration follows the standard ML pipeline: **Forward Pass → Error → Loss (MSE) → Backward
Pass (chain rule, m₁/m₂/b₁/b₂ gradients shown separately) → Parameter Update.**

| Iter | m1 | m2 | b1 | b2 | J (MSE) |
|---|---|---|---|---|---|
| 0 | -1.0000 | 2.0000 | 1.00000 | 1.00000 | 61.00 |
| 1 | -1.4500 | 0.8700 | 0.99000 | 0.89000 | 61.000 |
| 2 | -1.3331 | 1.1765 | 1.01850 | 0.91210 | 6.5033 |
| 3 | -1.369037 | 1.095583 | 1.036351 | 0.898653 | 2.4974 |
| 4 | -1.363711 | 1.119128 | 1.056810 | 0.894870 | 2.164498 |

**Key point for Q&A:** b₁ only ever affects point 1's prediction and b₂ only ever affects point 2's
— so unlike m₁/m₂ (whose gradients sum over both points), each bᵢ's gradient is a single term, not
a sum. That's why b₁ drifts up (~1.06) while b₂ drifts down (~0.89) instead of moving together.

**Trend:** J decreases every iteration (61.0 → 6.50 → 2.50 → 2.16) — confirms gradient descent is
moving in the correct direction. Full step-by-step arithmetic (including every chain rule expansion)
is in `part3_manual_gradient_descent.pdf` — transcribe it by hand for your submission.

---

## Part 4 — Gradient Descent in Code (SciPy + Matplotlib)

Uses SciPy's `approx_fprime` to numerically differentiate the MSE cost function — cross-verified
against the manual Part 3 derivation:

| Iter | SciPy dJ/dm1 | Manual ∂J/∂m1 | SciPy dJ/db1 | Manual ∂J/∂b1 |
|---|---|---|---|---|
| 1 | 45.000009 | 45.0 | 1.000001 | 1.0 |
| 2 | -11.690016 | -11.69 | -2.850001 | -2.85 |
| 3 | 3.593690 | 3.5937 | -1.785101 | -1.7851 |
| 4 | -0.532608 | -0.532597 | -2.045938 | -2.045937 |

Matches to 4+ decimal places — good cross-validation moment to mention live.

Ran for 50 iterations total to show full convergence beyond the 4 manual steps:
- Final params: m ≈ [-1.4904, 1.1637], b ≈ [1.7795, 0.6825]
- Final cost: 0.808 (down from 61.0 at initialization)

### Convergence Plots

![Parameters over iterations](part4_parameters_over_iterations.png)

![Error over iterations](part4_error_over_iterations.png)

---

## Presentation Checklist (mapped to rubric)

- [ ] **EM Algorithm:** explain mean-split problem, E-step/M-step, log-likelihood stopping rule,
      show tracking table, run live classification with coach's height
- [ ] **Bayesian Probability:** state P(Positive|keyword) choice, justify keywords, walk through
      one full worked Bayes calculation live, ensure every member can discuss ≥1 keyword
- [ ] **Gradient Descent:** walk the PDF iteration-by-iteration, emphasize the explicit chain rule
      for ∂J/∂m AND ∂J/∂b (note b's independent per-point gradient), each member presents their
      assigned iteration
- [ ] Confirm every member can answer Q&A on their own section without notes
- [ ] Commit all code to your shared repo with per-member commit messages (contribution proof)
