# Formative 3 — Probability Distributions, Bayesian Probability & Gradient Descent

### Task Sheet: [Team Task Sheet_ML_F3_C3 _G10](https://docs.google.com/spreadsheets/d/1KHgaNNePcu52AQp5eCi_h_6w3lCj5YCcLVAN_V6VB7w/edit?usp=sharing)

**Group:** 4 members. **Datasets:** `GaltonFamilies.csv` (Part 1, included),
`IMDB_Dataset.csv` (Part 2, 50k reviews — download link below, too large to commit raw).

## Files

- `em_heights.py` — Part 1: EM algorithm on pooled father/children heights
- `GaltonFamilies.csv` — dataset for Part 1
- `part1_means_over_iterations.png`, `part1_loglikelihood_over_iterations.png` — Part 1 plots
- `part2_bayesian_sentiment.py` — Part 2: Bayes' Theorem on IMDb reviews (pure Python)
- `part3_manual_gradient_descent.pdf` — Part 3: 4 handwritten/typed GD iterations
- `part4_gradient_descent_code.py` — Part 4: same GD problem in code, SciPy-verified
- `part4_parameters_over_iterations.png`, `part4_error_over_iterations.png` — Part 4 plots

## Run it

```bash
python3 em_heights.py                 # needs GaltonFamilies.csv in same folder
python3 part2_bayesian_sentiment.py   # needs IMDB_Dataset.csv in same folder
python3 part4_gradient_descent_code.py
```

IMDb dataset (50k reviews): https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv
— download and save as `IMDB_Dataset.csv`.

## Key results

- **Part 1:** EM converges to μ1=64.74" (children), μ2=69.68" (fathers) vs. a
  naive mean-split giving 64.26"/70.21" — split-at-the-mean biases both
  groups because the distributions overlap.
- **Part 2:** every positive keyword pushes P(Positive|keyword) above the
  0.50 prior (up to 0.81 for "excellent"); every negative keyword pulls it
  below (down to 0.093 for "worst").
- **Part 3/4:** cost J falls every iteration (61.0 → 6.50 → 2.49 → 2.18),
  and hand-derived gradients match SciPy's numerical gradient to 4 decimals.
