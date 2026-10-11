from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from datasets.models import Dataset, DatasetStatus
from reviews.models import SentimentResult

User = get_user_model()


class DatasetIngestionTests(TestCase):
    def setUp(self):
        self.user1 = User.objects.create_user(username="user1", password="Password123!")
        self.user2 = User.objects.create_user(username="user2", password="Password123!")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user1)

    def test_upload_valid_csv(self):
        csv_content = (
            b"review_id,review_text,rating,review_date,product_category\n"
            b"R101,Awesome product! Really loved it.,5,2026-09-01,Electronics\n"
            b"R102,Arrived on time and works as described.,3,2026-09-02,Home\n"
            b"R103,Broken on arrival and support was unhelpful.,1,2026-09-03,Electronics\n"
        )
        file = SimpleUploadedFile("reviews.csv", csv_content, content_type="text/csv")

        response = self.client.post(
            "/api/datasets/",
            {"file": file, "name": "Customer Feedback Q3"},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        dataset_id = response.data["dataset_id"]

        dataset = Dataset.objects.get(id=dataset_id)
        self.assertEqual(dataset.status, DatasetStatus.COMPLETED)
        self.assertEqual(dataset.valid_rows, 3)
        self.assertEqual(dataset.invalid_rows, 0)
        self.assertEqual(dataset.reviews.count(), 3)
        self.assertEqual(SentimentResult.objects.filter(review__dataset=dataset).count(), 3)

        # Check prediction labels
        pos_review = dataset.reviews.get(external_review_id="R101")
        self.assertEqual(pos_review.sentiment.label, "POSITIVE")
        neg_review = dataset.reviews.get(external_review_id="R103")
        self.assertEqual(neg_review.sentiment.label, "NEGATIVE")

    def test_reject_missing_review_text_column(self):
        csv_content = b"id,stars,date\n1,5,2026-09-01\n2,4,2026-09-02\n"
        file = SimpleUploadedFile("invalid_columns.csv", csv_content, content_type="text/csv")

        response = self.client.post(
            "/api/datasets/",
            {"file": file, "name": "Invalid Dataset"},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Missing required review text column", response.data["error"])

    def test_reject_non_csv_extension(self):
        file = SimpleUploadedFile("notes.txt", b"some notes", content_type="text/plain")
        response = self.client.post(
            "/api/datasets/",
            {"file": file},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_handles_invalid_and_duplicate_rows(self):
        csv_content = (
            b"review_id,feedback,rating,date\n"
            b"D1,Great service,5,2026-09-01\n"
            b"D1,Duplicate ID should be rejected,4,2026-09-02\n"  # Duplicate id
            b"D2,   ,2,2026-09-03\n"  # Empty review
            b"D3,Acceptable quality,100,2026-09-04\n"  # Rating out of bounds -> clamped/null
        )
        file = SimpleUploadedFile("dirty.csv", csv_content, content_type="text/csv")

        response = self.client.post("/api/datasets/", {"file": file}, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

        dataset = Dataset.objects.get(id=response.data["dataset_id"])
        self.assertEqual(dataset.valid_rows, 2)  # D1 and D3
        self.assertEqual(dataset.invalid_rows, 2)  # duplicate D1 and empty D2
        self.assertEqual(dataset.reviews.count(), 2)

    def test_ownership_authorization(self):
        # Create dataset for user1
        csv_content = b"review_text\nAwesome service\n"
        file = SimpleUploadedFile("test.csv", csv_content, content_type="text/csv")
        upload_resp = self.client.post("/api/datasets/", {"file": file}, format="multipart")
        dataset_id = upload_resp.data["dataset_id"]

        # User2 tries to access user1's dataset
        self.client.force_authenticate(user=self.user2)
        get_resp = self.client.get(f"/api/datasets/{dataset_id}/")
        self.assertEqual(get_resp.status_code, status.HTTP_403_FORBIDDEN)

        status_resp = self.client.get(f"/api/datasets/{dataset_id}/status/")
        self.assertEqual(status_resp.status_code, status.HTTP_403_FORBIDDEN)

        del_resp = self.client.delete(f"/api/datasets/{dataset_id}/")
        self.assertEqual(del_resp.status_code, status.HTTP_403_FORBIDDEN)

    def test_retry_idempotency(self):
        csv_content = b"review_id,review_text\nR1,Good product\nR2,Bad product\n"
        file = SimpleUploadedFile("retry_test.csv", csv_content, content_type="text/csv")
        upload_resp = self.client.post("/api/datasets/", {"file": file}, format="multipart")
        dataset_id = upload_resp.data["dataset_id"]

        dataset = Dataset.objects.get(id=dataset_id)
        self.assertEqual(dataset.reviews.count(), 2)

        # Retry the dataset
        retry_resp = self.client.post(f"/api/datasets/{dataset_id}/retry/")
        self.assertEqual(retry_resp.status_code, status.HTTP_202_ACCEPTED)

        dataset.refresh_from_db()
        self.assertEqual(dataset.status, DatasetStatus.COMPLETED)
        # Verify no duplicate reviews were created
        self.assertEqual(dataset.reviews.count(), 2)
        self.assertEqual(SentimentResult.objects.filter(review__dataset=dataset).count(), 2)
