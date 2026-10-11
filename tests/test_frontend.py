from django.contrib.auth import get_user_model
from django.test import TestCase

from datasets.models import Dataset, DatasetStatus

User = get_user_model()


class FrontendViewsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testui", password="Password123!")
        self.dataset = Dataset.objects.create(
            owner=self.user,
            name="UI Test Dataset",
            original_filename="reviews.csv",
            status=DatasetStatus.COMPLETED,
            total_rows=10,
            valid_rows=10,
        )

    def test_anonymous_redirects(self):
        response = self.client.get("/dashboard/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)

    def test_authenticated_dashboard(self):
        self.client.force_login(self.user)
        response = self.client.get("/dashboard/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Feedback Datasets")
        self.assertContains(response, "UI Test Dataset")

    def test_upload_page(self):
        self.client.force_login(self.user)
        response = self.client.get("/datasets/upload/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Upload Review Dataset")

    def test_analytics_dashboard(self):
        self.client.force_login(self.user)
        response = self.client.get(f"/datasets/{self.dataset.id}/analytics/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "UI Test Dataset")
        self.assertContains(response, "Sentiment Distribution")

    def test_review_explorer_page(self):
        self.client.force_login(self.user)
        response = self.client.get(f"/datasets/{self.dataset.id}/reviews/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Review Explorer")

    def test_dataset_details_page(self):
        self.client.force_login(self.user)
        response = self.client.get(f"/datasets/{self.dataset.id}/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Dataset Details")
        self.assertContains(response, "Ingestion Validation Report")

    def test_model_info_page(self):
        # Publicly accessible
        response = self.client.get("/model-info/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Sentiment Classification Model")
        self.assertContains(response, "Confusion Matrix")
