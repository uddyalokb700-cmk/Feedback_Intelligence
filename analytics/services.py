"""
Analytics computation service using database-level SQL aggregations.
Computes summary KPIs, sentiment distribution, time series trends, category metrics, and top terms.
"""

import re
from collections import Counter

from django.core.cache import cache
from django.db.models import Avg, Count, Q

from datasets.models import Dataset
from reviews.models import Review

CACHE_TTL = 300  # 5 minutes


# Basic English stopwords for top terms extraction
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "item", "product", "one", "just", "like", "get"
}


def get_summary_metrics(dataset: Dataset, use_cache: bool = True) -> dict:
    """
    Compute core summary KPIs using pure database aggregation queries.
    """
    cache_key = f"cfi_summary_{dataset.id}_{dataset.completed_at}"
    if use_cache:
        cached = cache.get(cache_key)
        if cached:
            return cached

    reviews_qs = Review.objects.filter(dataset=dataset)
    total_valid = reviews_qs.count()

    if total_valid == 0:
        return {
            "total_valid_reviews": 0,
            "total_invalid_rows": dataset.invalid_rows,
            "average_rating": None,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
            "positive_percentage": 0.0,
            "neutral_percentage": 0.0,
            "negative_percentage": 0.0,
            "negative_review_rate": 0.0,
            "low_rating_rate": None,
            "rating_distribution": {},
        }

    # Aggregate counts and average rating in a single SQL query
    agg = reviews_qs.aggregate(
        avg_rating=Avg("rating"),
        total_rated=Count("rating"),
        low_rated=Count("rating", filter=Q(rating__lte=2.0)),
        pos_count=Count("sentiment", filter=Q(sentiment__label="POSITIVE")),
        neu_count=Count("sentiment", filter=Q(sentiment__label="NEUTRAL")),
        neg_count=Count("sentiment", filter=Q(sentiment__label="NEGATIVE")),
    )

    pos_c = agg["pos_count"] or 0
    neu_c = agg["neu_count"] or 0
    neg_c = agg["neg_count"] or 0
    total_sent = pos_c + neu_c + neg_c

    # Percentages
    pos_pct = round((pos_c / total_sent) * 100, 2) if total_sent > 0 else 0.0
    neu_pct = round((neu_c / total_sent) * 100, 2) if total_sent > 0 else 0.0
    neg_pct = round((neg_c / total_sent) * 100, 2) if total_sent > 0 else 0.0
    neg_rate = round((neg_c / total_sent), 4) if total_sent > 0 else 0.0

    avg_r = round(agg["avg_rating"], 2) if agg["avg_rating"] is not None else None
    total_rated = agg["total_rated"] or 0
    low_rate = round((agg["low_rated"] / total_rated) * 100, 2) if total_rated > 0 else None

    # Rating distribution (1 to 5 stars) if ratings exist
    rating_dist = {}
    if total_rated > 0:
        # Group by integer rounded rating
        rating_counts = (
            reviews_qs.filter(rating__isnull=False)
            .values("rating")
            .annotate(cnt=Count("id"))
            .order_by("rating")
        )
        for rc in rating_counts:
            r_key = str(int(rc["rating"])) if rc["rating"].is_integer() else f"{rc['rating']:.1f}"
            rating_dist[r_key] = rating_dist.get(r_key, 0) + rc["cnt"]

    result = {
        "total_valid_reviews": total_valid,
        "total_invalid_rows": dataset.invalid_rows,
        "average_rating": avg_r,
        "positive_count": pos_c,
        "neutral_count": neu_c,
        "negative_count": neg_c,
        "positive_percentage": pos_pct,
        "neutral_percentage": neu_pct,
        "negative_percentage": neg_pct,
        "negative_review_rate": neg_rate,
        "low_rating_rate": low_rate,
        "rating_distribution": rating_dist,
    }

    if use_cache:
        cache.set(cache_key, result, timeout=CACHE_TTL)
    return result


def get_sentiment_distribution(dataset: Dataset) -> dict:
    """
    Counts and proportions for POSITIVE, NEUTRAL, and NEGATIVE labels.
    """
    summary = get_summary_metrics(dataset)
    return {
        "positive": {
            "count": summary["positive_count"],
            "percentage": summary["positive_percentage"],
        },
        "neutral": {
            "count": summary["neutral_count"],
            "percentage": summary["neutral_percentage"],
        },
        "negative": {
            "count": summary["negative_count"],
            "percentage": summary["negative_percentage"],
        },
        "total_classified": summary["positive_count"] + summary["neutral_count"] + summary["negative_count"],
    }


def get_trends(dataset: Dataset) -> dict:
    """
    Aggregate review volume, sentiment count, and average rating grouped by date.
    Uses database-level date truncation.
    """
    cache_key = f"cfi_trends_{dataset.id}_{dataset.completed_at}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    # Group by review_date
    date_records = (
        Review.objects.filter(dataset=dataset, review_date__isnull=False)
        .values("review_date")
        .annotate(
            total_reviews=Count("id"),
            avg_rating=Avg("rating"),
            positive=Count("sentiment", filter=Q(sentiment__label="POSITIVE")),
            neutral=Count("sentiment", filter=Q(sentiment__label="NEUTRAL")),
            negative=Count("sentiment", filter=Q(sentiment__label="NEGATIVE")),
        )
        .order_by("review_date")
    )

    labels = []
    volume_data = []
    avg_rating_data = []
    pos_data = []
    neu_data = []
    neg_data = []

    for r in date_records:
        date_str = r["review_date"].strftime("%Y-%m-%d")
        labels.append(date_str)
        volume_data.append(r["total_reviews"])
        avg_rating_data.append(round(r["avg_rating"], 2) if r["avg_rating"] is not None else None)
        pos_data.append(r["positive"])
        neu_data.append(r["neutral"])
        neg_data.append(r["negative"])

    trends = {
        "dates": labels,
        "volume": volume_data,
        "average_rating": avg_rating_data,
        "positive": pos_data,
        "neutral": neu_data,
        "negative": neg_data,
    }

    cache.set(cache_key, trends, timeout=CACHE_TTL)
    return trends


def get_category_metrics(dataset: Dataset) -> list[dict]:
    """
    Compute volume, average rating, and sentiment proportion for each product category.
    """
    cache_key = f"cfi_cats_{dataset.id}_{dataset.completed_at}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    cat_records = (
        Review.objects.filter(dataset=dataset, product_category__isnull=False)
        .exclude(product_category="")
        .values("product_category")
        .annotate(
            review_count=Count("id"),
            avg_rating=Avg("rating"),
            positive=Count("sentiment", filter=Q(sentiment__label="POSITIVE")),
            neutral=Count("sentiment", filter=Q(sentiment__label="NEUTRAL")),
            negative=Count("sentiment", filter=Q(sentiment__label="NEGATIVE")),
        )
        .order_by("-review_count")
    )

    categories = []
    for c in cat_records:
        total = c["review_count"]
        pos = c["positive"]
        neg = c["negative"]
        neu = c["neutral"]
        classified = pos + neu + neg
        categories.append({
            "category": c["product_category"],
            "review_count": total,
            "average_rating": round(c["avg_rating"], 2) if c["avg_rating"] is not None else None,
            "positive_count": pos,
            "neutral_count": neu,
            "negative_count": neg,
            "positive_percentage": round((pos / classified) * 100, 1) if classified > 0 else 0.0,
            "negative_review_rate": round(neg / classified, 4) if classified > 0 else 0.0,
        })

    cache.set(cache_key, categories, timeout=CACHE_TTL)
    return categories


def get_top_terms(dataset: Dataset, limit: int = 15) -> dict:
    """
    Extract top recurring keywords from positive and negative reviews.
    Uses regex tokenization and stopword removal.
    """
    cache_key = f"cfi_top_terms_{dataset.id}_{dataset.completed_at}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    # Limit to bounded sample of reviews if very large (e.g. up to 1000 per class)
    pos_reviews = (
        Review.objects.filter(dataset=dataset, sentiment__label="POSITIVE")
        .values_list("review_text", flat=True)[:1000]
    )
    neg_reviews = (
        Review.objects.filter(dataset=dataset, sentiment__label="NEGATIVE")
        .values_list("review_text", flat=True)[:1000]
    )

    def extract_keywords(texts):
        counter = Counter()
        word_re = re.compile(r"\b[a-zA-Z]{3,15}\b")
        for text in texts:
            words = word_re.findall(text.lower())
            for w in words:
                if w not in STOPWORDS:
                    counter[w] += 1
        return [{"term": term, "count": cnt} for term, cnt in counter.most_common(limit)]

    result = {
        "positive_terms": extract_keywords(pos_reviews),
        "negative_terms": extract_keywords(neg_reviews),
    }

    cache.set(cache_key, result, timeout=CACHE_TTL)
    return result
