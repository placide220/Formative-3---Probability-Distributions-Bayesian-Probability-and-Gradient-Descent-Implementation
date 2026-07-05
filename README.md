# Formative 3: Probability Distributions, Bayesian Probability, and Gradient Descent

This repository contains one combined Jupyter Notebook for the full Formative 3 assignment. It covers:

1. Probability distributions with an EM algorithm on Galton family heights
2. Bayesian probability on IMDb movie reviews
3. Manual-gradient-descent support values
4. Gradient descent in Python using SciPy and Matplotlib

## Members

- Kevin Nizeyimana
- UWENAYO Alain Paicifique
- Placide Niyonizeye
- Benitha Iradukunda

## Task Sheet

- [Google Sheets task sheet](https://docs.google.com/spreadsheets/d/1KHgaNNePcu52AQp5eCi_h_6w3lCj5YCcLVAN_V6VB7w/edit?usp=sharing)

## Main Files

- `Formative_3_Probability_Distributions,_Bayesian_Probability,_and_Gradient_Descent_Implementation.ipynb`  
  Clean source notebook.
- `Formative_3_Probability_Distributions_Executed.ipynb`  
  Final executed notebook with outputs, tables, and plots.
- `part1_means_and_loglikelihood.png`  
  EM mean and log-likelihood plots.
- `part2_bayesian_sentiment_keywords.png`  
  Bayesian sentiment posterior plot.
- `manual_calculations.pdf`  
  Part 3 manual gradient descent calculations.
- `part4_gradient_descent_results.png`  
  Gradient descent parameter and cost plots.

The raw datasets are **not committed** to this repository because the IMDb CSV is large. Download them from Kaggle:

- Galton parent/child heights: https://www.kaggle.com/datasets/jacopoferretti/parents-heights-vs-children-heights-galton-data
- IMDb 50,000 movie reviews: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

After downloading, place either the CSV files or the zip files in the project root before running the source notebook.

Expected file names:

- `GaltonFamilies.csv`, or `parents-heights-vs-children-heights-galton-data.zip`
- `IMDB Dataset.csv`, or `imdb-dataset-of-50k-movie-reviews.zip`

The submitted executed notebook already contains the computed outputs, so the raw datasets are only needed if you want to re-run the notebook from scratch.

## Part 1: EM Algorithm

We compare **children heights** and **father heights** from the Galton dataset. The labels are hidden from the EM algorithm by pooling both groups into one unlabeled height array.

The EM model fits two Gaussian distributions using:

- Means: `mu1`, `mu2`
- Variances: `sigma1_2`, `sigma2_2`
- Mixing coefficients: `pi1`, `pi2`
- Log-likelihood to track convergence

### Required Tracking Table

The executed notebook prints the full table. The first required rows are:

| Iteration | mu1 Children | mu2 Fathers | sigma1^2 | sigma2^2 | pi1 | pi2 | Log-Likelihood |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64.5000 | 70.0000 | 12.7241 | 12.7241 | 0.5000 | 0.5000 | -3119.8222 |
| 1 | 65.2205 | 69.2054 | 9.0603 | 8.4468 | 0.5011 | 0.4989 | -3057.9135 |
| 2 | 65.1609 | 69.2600 | 8.9420 | 8.1040 | 0.5005 | 0.4995 | -3056.9719 |

Final EM parameters after 10 iterations:

- `mu1 = 64.7354`
- `mu2 = 69.6765`
- `sigma1_2 = 7.0185`
- `sigma2_2 = 6.2236`
- `pi1 = 0.4995`
- `pi2 = 0.5005`

### Mean-Split Discussion

We should **not** simply draw a line at the global mean and split the data into two piles.

The global mean is `67.21`. A hard split gives:

- Lower group mean: `64.26`
- Upper group mean: `70.21`

This is weaker than EM because child and father heights overlap. A hard cutoff forces each height into one group even when the height is uncertain. EM uses soft assignments, so each point can partially belong to both Gaussian components.

### Posterior Classification Demo

For a test height of `68` inches, the executed notebook outputs:

- `P(child) = 0.3553`
- `P(father) = 0.6447`

During presentation, change the test height in `classify_height(...)` to the value given by the coach and re-run that cell.

## Part 2: Bayesian Probability

We compute only:

```text
P(Positive | keyword)
```

This matches the assignment requirement to choose one conditional probability instead of computing both positive and negative posteriors.

Selected positive keywords:

- `excellent`
- `brilliant`
- `masterpiece`
- `love`

Selected negative keywords:

- `boring`
- `waste`
- `awful`
- `worst`

The implementation uses basic Python counting and Bayes' Theorem:

```text
P(Positive | keyword) = P(keyword | Positive) * P(Positive) / P(keyword)
```

Summary of executed results:

| Keyword | P(Positive) | P(keyword\|Positive) | P(keyword) | P(Positive\|keyword) |
|---|---:|---:|---:|---:|
| excellent | 0.5000 | 0.1147 | 0.0710 | 0.8074 |
| brilliant | 0.5000 | 0.0635 | 0.0418 | 0.7601 |
| masterpiece | 0.5000 | 0.0351 | 0.0241 | 0.7274 |
| love | 0.5000 | 0.2266 | 0.1785 | 0.6349 |
| boring | 0.5000 | 0.0236 | 0.0610 | 0.1937 |
| waste | 0.5000 | 0.0070 | 0.0507 | 0.0691 |
| awful | 0.5000 | 0.0114 | 0.0577 | 0.0985 |
| worst | 0.5000 | 0.0164 | 0.0887 | 0.0927 |

Positive keywords raise the posterior above the `0.50` prior. Negative keywords reduce it below the prior.

## Part 3: Manual Gradient Descent

The notebook includes the same setup used for the manual calculation:

```text
m = [-1, 2]
b = [1, 1]
X = [[1, 3], [4, 10]]
y = [5, 6]
alpha = 0.01
```

The Part 3 manual work is provided in `manual_calculations.pdf`. It shows:

- Predictions using matrix multiplication
- MSE cost
- Chain rule for `dJ/dm`
- Chain rule for `dJ/db`
- At least one complete update per group member
- Intermediate values after each update

## Part 4: Gradient Descent in Code

The notebook converts the manual calculation into Python. SciPy's `approx_fprime` is used to compute numerical derivatives for the cost function.

The first updates from the executed notebook are:

| Iteration | m1 | m2 | b1 | b2 | Cost |
|---:|---:|---:|---:|---:|---:|
| 0 | -1.000000 | 2.000000 | 1.000000 | 1.000000 | 61.000000 |
| 1 | -1.450000 | 0.869999 | 0.990000 | 0.890000 | 6.503318 |
| 2 | -1.333100 | 1.176500 | 1.018500 | 0.912100 | 2.497397 |
| 3 | -1.369037 | 1.095582 | 1.036351 | 0.898653 | 2.164499 |
| 4 | -1.363711 | 1.119127 | 1.056810 | 0.894870 | 2.099581 |

After 50 iterations:

- `m1 = -1.490421`
- `m2 = 1.163681`
- `b1 = 1.779502`
- `b2 = 0.682520`
- Cost: `0.808004`
- Final predictions: `[3.780123, 6.357643]`

The error decreases strongly from `61.000000` to `0.808004`, showing that the parameters move in a direction that reduces the MSE.
