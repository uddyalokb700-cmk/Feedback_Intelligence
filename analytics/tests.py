import datetime

from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from datasets.models import Dataset, DatasetStatus
from reviews.models import Review, SentimentResult

User = get_user_model()


class AnalyticsServiceTests(TestCase):
    def setUp(self):
        self.user1 = User.objects.create_user(username="u1", password="Password123!")
        self.user2 = User.objects.create_user(username="u2", password="Password123!")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user1)

        self.dataset = Dataset.objects.create(
            owner=self.user1,
            name="Analytics Test Dataset",
            original_filename="reviews.csv",
            status=DatasetStatus.COMPLETED,
            total_rows=6,
            valid_rows=5,
            invalid_rows=1,
        )

        # 2 Positive, 1 Neutral, 2 Negative
        data = [
            ("Great battery life and fast shipping", 5.0, "2026-09-01", "Electronics", "POSITIVE"),
            ("Super happy with quality and price", 4.0, "2026-09-01", "Electronics", "POSITIVE"),
            ("Average item, works okay", 3.0, "2026-09-02", "Home", "NEUTRAL"),
            ("Terrible shipping delay, arrived broken", 1.0, "2026-09-03", "Home", "NEGATIVE"),
            ("Horrible customer support, rude agent", 2.0, "2026-09-03", "Services", "NEGATIVE"),
        ]

        for text, rating, d_str, cat, label in data:
            dt = datetime.datetime.strptime(d_str, "%Y-%m-%d").date()
            r = Review.objects.create(
                dataset=self.dataset,
                review_text=text,
                rating=rating,
                review_date=dt,
                product_category=cat,
            )
            SentimentResult.objects.create(
                review=r,
                label=label,
                confidence=0.9,
                model_version="v1.0-tfidf-logreg",
            )

    def test_summary_kpis(self):
        response = self.client.get(f"/api/datasets/{self.dataset.id}/summary/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.data

        self.assertEqual(data["total_valid_reviews"], 5)
        self.assertEqual(data["total_invalid_rows"], 1)
        self.assertEqual(data["positive_count"], 2)
        self.assertEqual(data["neutral_count"], 1)
        self.assertEqual(data["negative_count"], 2)

        # Average rating: (5 + 4 + 3 + 1 + 2) / 5 = 3.0
        self.assertEqual(data["average_rating"], 3.0)

        # Negative review rate: 2 / 5 = 0.4
        self.assertEqual(data["negative_review_rate"], 0.4)

        # Low rating rate: 2 ratings <= 2.0 -> 40.0%
        self.assertEqual(data["low_rating_rate"], 40.0)

    def test_sentiment_distribution(self):
        response = self.client.get(f"/api/datasets/{self.dataset.id}/sentiment-distribution/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        dist = response.data
        self.assertEqual(dist["positive"]["count"], 2)
        self.assertEqual(dist["neutral"]["count"], 1)
        self.assertEqual(dist["negative"]["count"], 2)
        self.assertEqual(dist["total_classified"], 5)

    def test_trends_endpoint(self):
        response = self.client.get(f"/api/datasets/{self.dataset.id}/trends/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        trends = response.data
        self.assertIn("dates", trends)
        self.assertIn("volume", trends)
        self.assertIn("positive", trends)
        self.assertEqual(len(trends["dates"]), 3)  # 2026-09-01, 2026-09-02, 2026-09-03

    def test_category_metrics(self):
        response = self.client.get(f"/api/datasets/{self.dataset.id}/categories/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        cats = response.data
        self.assertEqual(len(cats), 3)
        cat_names = [c["category"] for c in cats]
        self.assertIn("Electronics", cat_names)
        self.assertIn("Home", cat_names)
        self.assertIn("Services", cat_names)

    def test_top_terms(self):
        response = self.client.get(f"/api/datasets/{self.dataset.id}/top-terms/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        terms = response.data
        self.assertIn("positive_terms", terms)
        self.assertIn("negative_terms", terms)

    def test_cross_user_analytics_denied(self):
        self.client.force_authenticate(user=self.user2)
        response = self.client.get(f"/api/datasets/{self.dataset.id}/summary/")
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
