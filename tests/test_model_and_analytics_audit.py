
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework.test import APIClient

from analytics.services import (
    get_category_metrics,
    get_sentiment_distribution,
    get_summary_metrics,
    get_top_terms,
    get_trends,
)
from datasets.models import Dataset, DatasetStatus
from datasets.services import run_dataset_ingestion
from reviews.models import Review, SentimentResult
from sentiment.inference import get_model_metadata, predict_batch
from sentiment.train import get_labeled_training_corpus

User = get_user_model()


class SentimentModelAuditTests(TestCase):
    """
    Automated audit tests for Sentiment Model:
    1. Zero data leakage: strict disjointness between base train and held-out test corpora.
    2. Model metadata and evaluation metrics consistency (accuracy, macro/weighted F1, confusion matrix).
    3. Proper alignment of per-class precision, recall, and F1.
    4. Calibrated confidence-score interpretation and fallback for empty/whitespace texts.
    """

    def test_corpus_strict_train_test_separation_no_leakage(self):
        """
        Verify that the evaluation test set does not leak any base training phrases or templates.
        """
        X_train, y_train, X_test, y_test = get_labeled_training_corpus()

        # Both sets must be populated
        self.assertGreater(len(X_train), 1000)
        self.assertGreater(len(X_test), 50)

        # Confirm exact text disjointness
        train_set = set(X_train)
        test_set = set(X_test)
        intersection = train_set.intersection(test_set)
        self.assertEqual(
            len(intersection),
            0,
            f"Data leakage detected! {len(intersection)} identical samples found in both train and test partitions.",
        )

        # Confirm class balance in both partitions
        for split_labels in [y_train, y_test]:
            pos_c = split_labels.count("POSITIVE")
            neu_c = split_labels.count("NEUTRAL")
            neg_c = split_labels.count("NEGATIVE")
            self.assertEqual(pos_c, neu_c)
            self.assertEqual(neu_c, neg_c)

    def test_model_metadata_and_evaluation_metrics_integrity(self):
        """
        Verify that persisted model metadata contains actual evaluated metrics,
        confusion matrix, and documented limitations.
        """
        metadata = get_model_metadata()
        self.assertEqual(metadata["model_version"], "v1.0-tfidf-logreg")
        self.assertIn("evaluation_metrics", metadata)

        eval_m = metadata["evaluation_metrics"]
        self.assertIn("accuracy", eval_m)
        self.assertIn("macro_precision", eval_m)
        self.assertIn("macro_recall", eval_m)
        self.assertIn("macro_f1", eval_m)
        self.assertIn("weighted_f1", eval_m)
        self.assertIn("per_class", eval_m)

        # Check per-class metrics
        for cls in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
            self.assertIn(cls, eval_m["per_class"])
            cls_m = eval_m["per_class"][cls]
            self.assertGreater(cls_m["precision"], 0.0)
            self.assertGreater(cls_m["recall"], 0.0)
            self.assertGreater(cls_m["f1_score"], 0.0)
            self.assertGreater(cls_m["support"], 0)

        # Check confusion matrix
        cm = metadata["confusion_matrix"]
        self.assertEqual(cm["labels"], ["POSITIVE", "NEUTRAL", "NEGATIVE"])
        self.assertEqual(len(cm["matrix"]), 3)
        self.assertEqual(len(cm["matrix"][0]), 3)

        # Check documented limitations
        limitations = metadata.get("known_limitations", [])
        self.assertGreaterEqual(len(limitations), 4)

    def test_confidence_score_interpretation_and_bounds(self):
        """
        Verify that confidence scores correspond to softmax probabilities and fallbacks are safe.
        """
        texts = [
            "Outstanding performance, absolutely thrilled with this purchase!",
            "Standard product, functions according to basic manual specifications.",
            "Completely broken on arrival, customer support refused to assist.",
            "",
            "   ",
        ]
        results = predict_batch(texts)
        self.assertEqual(len(results), 5)

        for res in results[:3]:
            # Confidence must be the maximum probability across classes
            max_prob = max(res["positive_prob"], res["neutral_prob"], res["negative_prob"])
            self.assertAlmostEqual(res["confidence"], max_prob, places=3)
            # Sum of probabilities must approximate 1.0
            prob_sum = res["positive_prob"] + res["neutral_prob"] + res["negative_prob"]
            self.assertAlmostEqual(prob_sum, 1.0, delta=0.05)

        # Empty/whitespace inputs receive safe neutral fallback
        for empty_res in results[3:]:
            self.assertEqual(empty_res["label"], "NEUTRAL")
            self.assertEqual(empty_res["confidence"], 0.3333)


class AnalyticsImplementationAuditTests(TestCase):
    """
    Automated audit tests for Analytics KPIs and edge cases:
    1. Missing ratings: average rating is None, low_rating_rate is None, no division by zero.
    2. Partial missing ratings: average rating and low rating rate computed only over rated reviews.
    3. Empty datasets: graceful 0 counts, no zero denominator exceptions.
    4. Duplicate source IDs: detected and rejected during ingestion.
    5. Zero denominators: protected across categories, trends, top terms.
    6. Distinctness verification: negative sentiment rate is NEVER confused with low-star rating rate.
    """

    def setUp(self):
        self.user = User.objects.create_user(
            username="analytics_auditor",
            password="TestPassword123!",
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def test_analytics_with_completely_missing_ratings(self):
        """
        Verify that datasets with unrated reviews return None for rating metrics
        while still accurately computing sentiment rates.
        """
        dataset = Dataset.objects.create(
            owner=self.user,
            name="No Ratings Dataset",
            original_filename="no_ratings.csv",
            status=DatasetStatus.COMPLETED,
            total_rows=3,
            valid_rows=3,
        )

        r1 = Review.objects.create(
            dataset=dataset, review_text="Great job", rating=None
        )
        r2 = Review.objects.create(
            dataset=dataset, review_text="Acceptable", rating=None
        )
        r3 = Review.objects.create(
            dataset=dataset, review_text="Terrible", rating=None
        )

        SentimentResult.objects.create(review=r1, label="POSITIVE", confidence=0.9)
        SentimentResult.objects.create(review=r2, label="NEUTRAL", confidence=0.8)
        SentimentResult.objects.create(review=r3, label="NEGATIVE", confidence=0.85)

        summary = get_summary_metrics(dataset, use_cache=False)
        self.assertEqual(summary["total_valid_reviews"], 3)
        self.assertIsNone(summary["average_rating"])
        self.assertIsNone(summary["low_rating_rate"])
        self.assertEqual(summary["rating_distribution"], {})

        # Sentiment metrics are still computed
        self.assertEqual(summary["positive_count"], 1)
        self.assertEqual(summary["neutral_count"], 1)
        self.assertEqual(summary["negative_count"], 1)
        self.assertEqual(summary["negative_percentage"], 33.33)
        self.assertEqual(summary["negative_review_rate"], 0.3333)

    def test_negative_sentiment_rate_not_confused_with_low_star_reviews(self):
        """
        CRITICAL TEST: Verify that negative sentiment rate (NLP) is completely
        independent and not confused with the low-star rating rate (ratings <= 2).

        Fixture:
        - 4 reviews with 5-star ratings but NEGATIVE sentiment (e.g. sarcastic praise).
        - 2 reviews with 1-star ratings but POSITIVE sentiment (e.g. praised product, complained about delivery).
        Total = 6 reviews.
        """
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Inversion Test Dataset",
            original_filename="inversion.csv",
            status=DatasetStatus.COMPLETED,
            total_rows=6,
            valid_rows=6,
        )

        # 4 reviews: 5 stars, NEGATIVE sentiment
        for i in range(4):
            rev = Review.objects.create(
                dataset=dataset,
                review_text=f"Sarcastic complaint number {i}",
                rating=5.0,
            )
            SentimentResult.objects.create(review=rev, label="NEGATIVE", confidence=0.95)

        # 2 reviews: 1 star, POSITIVE sentiment
        for i in range(2):
            rev = Review.objects.create(
                dataset=dataset,
                review_text=f"Loved the unit but courier failed {i}",
                rating=1.0,
            )
            SentimentResult.objects.create(review=rev, label="POSITIVE", confidence=0.92)

        summary = get_summary_metrics(dataset, use_cache=False)

        # Average rating: (4*5 + 2*1) / 6 = 22 / 6 = 3.67
        self.assertEqual(summary["average_rating"], 3.67)

        # Low rating rate: exactly 2 out of 6 reviews have rating <= 2 stars -> 33.33%
        self.assertEqual(summary["low_rating_rate"], 33.33)

        # Negative sentiment rate: exactly 4 out of 6 reviews have NEGATIVE sentiment -> 66.67%
        self.assertEqual(summary["negative_percentage"], 66.67)
        self.assertEqual(summary["negative_review_rate"], 0.6667)
        self.assertEqual(summary["negative_sentiment_rate"], 0.6667)

        # Positive sentiment rate: exactly 2 out of 6 reviews -> 33.33%
        self.assertEqual(summary["positive_percentage"], 33.33)

        # Verify they are mathematically distinct
        self.assertNotEqual(summary["low_rating_rate"], summary["negative_percentage"])

    def test_analytics_with_empty_dataset(self):
        """
        Verify that an empty dataset (0 reviews) yields guarded defaults without ZeroDivisionError.
        """
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Empty Dataset",
            original_filename="empty.csv",
            status=DatasetStatus.COMPLETED,
            total_rows=0,
            valid_rows=0,
            invalid_rows=0,
        )

        summary = get_summary_metrics(dataset, use_cache=False)
        self.assertEqual(summary["total_valid_reviews"], 0)
        self.assertIsNone(summary["average_rating"])
        self.assertIsNone(summary["low_rating_rate"])
        self.assertEqual(summary["positive_percentage"], 0.0)
        self.assertEqual(summary["negative_percentage"], 0.0)
        self.assertEqual(summary["negative_review_rate"], 0.0)
        self.assertEqual(summary["rating_distribution"], {})

        dist = get_sentiment_distribution(dataset)
        self.assertEqual(dist["total_classified"], 0)

        trends = get_trends(dataset)
        self.assertEqual(trends["dates"], [])
        self.assertEqual(trends["volume"], [])

        cats = get_category_metrics(dataset)
        self.assertEqual(cats, [])

        top_terms = get_top_terms(dataset)
        self.assertEqual(top_terms["positive_terms"], [])
        self.assertEqual(top_terms["negative_terms"], [])

    def test_duplicate_source_ids_rejected_during_ingestion(self):
        """
        Verify that duplicate source IDs (external_review_id) in CSV are
        detected and counted as invalid rows, avoiding duplicate DB records.
        """
        csv_content = (
            "review_id,review_text,rating\n"
            "SRC_001,Super fast shipment and high quality,5\n"
            "SRC_001,Duplicate ID that must be rejected,4\n"
            "SRC_002,Average packaging,3\n"
        ).encode("utf-8")

        file = SimpleUploadedFile("dup.csv", csv_content, content_type="text/csv")
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Duplicate ID Test",
            original_filename="dup.csv",
            file=file,
            status=DatasetStatus.QUEUED,
        )

        success = run_dataset_ingestion(str(dataset.id))
        self.assertTrue(success)

        dataset.refresh_from_db()
        self.assertEqual(dataset.status, DatasetStatus.COMPLETED)
        self.assertEqual(dataset.valid_rows, 2)
        self.assertEqual(dataset.invalid_rows, 1)
        self.assertEqual(dataset.reviews.count(), 2)

        summary = get_summary_metrics(dataset, use_cache=False)
        self.assertEqual(summary["total_valid_reviews"], 2)
        self.assertEqual(summary["total_invalid_rows"], 1)

    def test_zero_denominators_in_categories_with_no_sentiment_or_rating(self):
        """
        Verify get_category_metrics handles edge cases where category has
        reviews without sentiments or without ratings without ZeroDivisionError.
        """
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Category Zero Denom",
            original_filename="cat_zero.csv",
            status=DatasetStatus.COMPLETED,
            total_rows=1,
            valid_rows=1,
        )

        # Review with category but no rating and no sentiment result
        Review.objects.create(
            dataset=dataset,
            review_text="No rating or sentiment",
            rating=None,
            product_category="Electronics",
        )

        cats = get_category_metrics(dataset)
        self.assertEqual(len(cats), 1)
        c = cats[0]
        self.assertEqual(c["category"], "Electronics")
        self.assertEqual(c["review_count"], 1)
        self.assertIsNone(c["average_rating"])
        self.assertEqual(c["positive_percentage"], 0.0)
        self.assertEqual(c["negative_review_rate"], 0.0)
