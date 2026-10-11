import datetime

from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from datasets.models import Dataset, DatasetStatus
from reviews.models import Review, SentimentResult

User = get_user_model()


class ReviewExplorerTests(TestCase):
    def setUp(self):
        self.user1 = User.objects.create_user(username="u1", password="Password123!")
        self.user2 = User.objects.create_user(username="u2", password="Password123!")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user1)

        self.dataset = Dataset.objects.create(
            owner=self.user1,
            name="Sample Reviews",
            original_filename="sample.csv",
            status=DatasetStatus.COMPLETED,
            total_rows=5,
            valid_rows=5,
        )

        # Create sample reviews with sentiments
        data = [
            ("E1", "Outstanding product, highly recommended!", 5.0, "2026-09-01", "Electronics", "POSITIVE"),
            ("E2", "Decent item, works as expected.", 3.0, "2026-09-02", "Electronics", "NEUTRAL"),
            ("E3", "Horrible quality, fell apart.", 1.0, "2026-09-03", "Home", "NEGATIVE"),
            ("E4", "Customer support was terrible.", 2.0, "2026-09-04", "Home", "NEGATIVE"),
            ("E5", "Love the battery life and display.", 5.0, "2026-09-05", "Electronics", "POSITIVE"),
        ]

        for ext_id, text, rating, d_str, cat, label in data:
            dt = datetime.datetime.strptime(d_str, "%Y-%m-%d").date()
            rev = Review.objects.create(
                dataset=self.dataset,
                external_review_id=ext_id,
                review_text=text,
                rating=rating,
                review_date=dt,
                product_category=cat,
            )
            SentimentResult.objects.create(
                review=rev,
                label=label,
                confidence=0.88,
                model_version="v1.0-tfidf-logreg",
            )

    def test_list_reviews_pagination(self):
        response = self.client.get(f"/api/datasets/{self.dataset.id}/reviews/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 5)
        self.assertEqual(len(response.data["results"]), 5)
        first = response.data["results"][0]
        self.assertIn("sentiment", first)
        self.assertIn("label", first["sentiment"])

    def test_filter_by_sentiment(self):
        response = self.client.get(f"/api/datasets/{self.dataset.id}/reviews/?sentiment=NEGATIVE")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 2)
        for r in response.data["results"]:
            self.assertEqual(r["sentiment"]["label"], "NEGATIVE")

    def test_filter_by_category_and_rating(self):
        response = self.client.get(f"/api/datasets/{self.dataset.id}/reviews/?category=Electronics&min_rating=4")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 2)

    def test_search_reviews(self):
        response = self.client.get(f"/api/datasets/{self.dataset.id}/reviews/?search=battery")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["external_review_id"], "E5")

    def test_export_reviews_csv(self):
        response = self.client.get(f"/api/datasets/{self.dataset.id}/reviews/export/?sentiment=POSITIVE")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response["Content-Type"], "text/csv")
        content = response.content.decode("utf-8")
        self.assertIn("review_text", content)
        self.assertIn("Outstanding product", content)
        self.assertNotIn("Horrible quality", content)

    def test_n_plus_one_avoidance(self):
        # 1 dataset lookup + 1 session user + 1 count + 1 joined reviews+sentiment query = 4 queries (constant)
        with self.assertNumQueries(4):
            response = self.client.get(f"/api/datasets/{self.dataset.id}/reviews/")
            self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_unauthorized_user_cannot_access_reviews(self):
        self.client.force_authenticate(user=self.user2)
        response = self.client.get(f"/api/datasets/{self.dataset.id}/reviews/")
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
