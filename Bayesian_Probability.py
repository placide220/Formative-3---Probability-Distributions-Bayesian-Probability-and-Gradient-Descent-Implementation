import csv
import os
import re


def load_data(csv_path="IMDB_Dataset.csv", text_col="review", label_col="sentiment"):
   
    reviews = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row[text_col].lower()
            label = row[label_col].strip().lower()
            reviews.append((text, label))
    return reviews



POSITIVE_KEYWORDS = ["excellent", "brilliant", "masterpiece", "love"]
NEGATIVE_KEYWORDS = ["boring", "waste", "awful", "worst"]
ALL_KEYWORDS = POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS


def contains_keyword(text, keyword):
    
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None



def bayes_for_keyword(reviews, keyword):
   
    total = len(reviews)
    n_positive = sum(1 for _, label in reviews if label == "positive")

    prior = n_positive / total

    pos_with_kw = sum(1 for text, label in reviews
                       if label == "positive" and contains_keyword(text, keyword))
    likelihood = pos_with_kw / n_positive if n_positive > 0 else 0.0

    n_with_kw = sum(1 for text, _ in reviews if contains_keyword(text, keyword))
    marginal = n_with_kw / total if total > 0 else 0.0

    if marginal == 0:
        posterior = 0.0
    else:
        # Bayes' Theorem:  P(Positive | keyword) = P(keyword | Positive) * P(Positive) / P(keyword)
        posterior = (likelihood * prior) / marginal

    return {
        "keyword": keyword,
        "prior": prior,
        "likelihood": likelihood,
        "marginal": marginal,
        "posterior": posterior,
        "n_with_kw": n_with_kw,
    }



def print_results_table(results):
    header = f"{'Keyword':<14}{'P(Positive)':<14}{'P(kw|Pos)':<14}{'P(kw)':<12}{'P(Pos|kw)':<12}{'n(kw)':<8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['keyword']:<14}{r['prior']:<14.4f}{r['likelihood']:<14.4f}"
              f"{r['marginal']:<12.4f}{r['posterior']:<12.4f}{r['n_with_kw']:<8}")


if __name__ == "__main__":
    
    reviews = load_data('/content/IMDB Dataset.csv')

    print(f"Loaded {len(reviews)} reviews "
          f"({sum(1 for _,l in reviews if l=='positive')} positive, "
          f"{sum(1 for _,l in reviews if l=='negative')} negative)\n")

    print("=== Keywords chosen as POSITIVE indicators ===")
    pos_results = [bayes_for_keyword(reviews, kw) for kw in POSITIVE_KEYWORDS]
    print_results_table(pos_results)

    print("\n=== Keywords chosen as NEGATIVE indicators ===")
    neg_results = [bayes_for_keyword(reviews, kw) for kw in NEGATIVE_KEYWORDS]
    print_results_table(neg_results)

    print("\nInterpretation: for positive-indicator keywords we expect P(Positive|keyword) "
          "to be well ABOVE the prior P(Positive); for negative-indicator keywords we expect "
          "P(Positive|keyword) to be well BELOW the prior -- discuss this contrast in your presentation.")
