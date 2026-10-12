import csv
import io
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from rest_framework import status
from rest_framework.test import APIClient

from analytics.services import clear_dataset_analytics_cache, get_summary_metrics, make_cache_key
from datasets.models import Dataset, DatasetStatus, JobStatus, ProcessingJob
from datasets.services import run_dataset_ingestion, validate_csv_structure
from reviews.models import Review, SentimentResult
from sentiment.inference import predict_batch

User = get_user_model()


class AuditPipelineRegressionTests(TestCase):
    """
    Regression test suite covering audit remediation items:
    1. Row-level concurrency locking preventing duplicate/concurrent processing.
    2. Atomic rollback & zero orphaned records on mid-stream ingestion failure.
    3. Bounded-memory row limit enforcement (MAX_CSV_ROWS).
    4. Flexible delimiter detection (semicolon, tab, pipe).
    5. High-throughput single-pass sentiment inference with sub-batching (>2,000 texts).
    6. Streaming HTTP response with bounded memory for CSV review exports.
    7. Cache invalidation on dataset lifecycle events and retries.
    """

    def setUp(self):
        self.user = User.objects.create_user(
            username="audit_tester",
            password="TestPassword123!",
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def test_concurrency_lock_rejects_duplicate_execution(self):
        """
        Verify that attempting to process an already PROCESSING dataset aborts
        immediately and cancels the duplicate job without corrupting state.
        """
        file = SimpleUploadedFile(
            "dummy.csv",
            b"review_text,rating\nGreat service,5\n",
            content_type="text/csv",
        )
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Concurrent Lock Test",
            original_filename="dummy.csv",
            file=file,
            status=DatasetStatus.PROCESSING,
        )
        duplicate_job = ProcessingJob.objects.create(
            dataset=dataset,
            status=JobStatus.PENDING,
        )

        success = run_dataset_ingestion(str(dataset.id), str(duplicate_job.id))

        self.assertFalse(success)
        duplicate_job.refresh_from_db()
        self.assertEqual(duplicate_job.status, JobStatus.CANCELLED)
        self.assertIn("duplicate", duplicate_job.safe_error_message.lower())

        dataset.refresh_from_db()
        self.assertEqual(dataset.status, DatasetStatus.PROCESSING)
        self.assertEqual(dataset.reviews.count(), 0)

    def test_partial_failure_cleans_up_records_atomically(self):
        """
        Verify that mid-stream failure purges partial reviews, leaving zero
        orphaned records, and updates job completion timestamps.
        """
        csv_lines = ["review_id,review_text,rating"]
        for i in range(10):
            csv_lines.append(f"R{i},Feedback item {i} is wonderful,5")
        csv_bytes = "\n".join(csv_lines).encode("utf-8")

        file = SimpleUploadedFile(
            "partial_fail.csv",
            csv_bytes,
            content_type="text/csv",
        )
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Partial Failure Cleanup",
            original_filename="partial_fail.csv",
            file=file,
            status=DatasetStatus.QUEUED,
        )
        job = ProcessingJob.objects.create(
            dataset=dataset,
            status=JobStatus.PENDING,
        )

        # Force failure during batch processing on second call
        call_count = 0
        from datasets import services as ds_services
        orig_process_batch = ds_services._process_review_batch

        def fail_on_second_batch(d, j, reviews, current_count):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise RuntimeError("Simulated mid-stream disk or network disconnect")
            return orig_process_batch(d, j, reviews, current_count)

        with override_settings(INGESTION_BATCH_SIZE=3):
            with patch.object(ds_services, "_process_review_batch", side_effect=fail_on_second_batch):
                result = run_dataset_ingestion(str(dataset.id), str(job.id))

        self.assertFalse(result)
        dataset.refresh_from_db()
        job.refresh_from_db()

        self.assertEqual(dataset.status, DatasetStatus.FAILED)
        self.assertIn("Simulated mid-stream", dataset.error_summary)
        self.assertEqual(dataset.processed_rows, 0)
        self.assertEqual(dataset.valid_rows, 0)
        self.assertIsNotNone(dataset.completed_at)

        # Crucial check: zero orphaned reviews in database
        self.assertEqual(Review.objects.filter(dataset=dataset).count(), 0)
        self.assertEqual(SentimentResult.objects.filter(review__dataset=dataset).count(), 0)

        # Job state consistency
        self.assertEqual(job.status, JobStatus.FAILED)
        self.assertIsNotNone(job.completed_at)
        self.assertEqual(job.error_code, "RuntimeError")

    def test_row_limit_enforcement(self):
        """
        Verify that CSV files exceeding MAX_CSV_ROWS are terminated before unbounded memory growth.
        """
        csv_lines = ["review_text,rating"]
        for i in range(8):
            csv_lines.append(f"Review {i} details here,4")
        csv_bytes = "\n".join(csv_lines).encode("utf-8")

        file = SimpleUploadedFile("exceeds_limit.csv", csv_bytes, content_type="text/csv")
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Row Limit Test",
            original_filename="exceeds_limit.csv",
            file=file,
            status=DatasetStatus.QUEUED,
        )

        with override_settings(MAX_CSV_ROWS=5):
            success = run_dataset_ingestion(str(dataset.id))

        self.assertFalse(success)
        dataset.refresh_from_db()
        self.assertEqual(dataset.status, DatasetStatus.FAILED)
        self.assertIn("exceeds maximum allowed limit", dataset.error_summary.lower())
        self.assertEqual(Review.objects.filter(dataset=dataset).count(), 0)

    def test_semicolon_delimited_csv_detection(self):
        """
        Verify that alternative delimiters (e.g. semicolon ';') are detected and parsed correctly.
        """
        csv_content = (
            "review_id;review_text;rating;product_category\n"
            "SC1;The battery lasts for several days;5;Hardware\n"
            "SC2;Frequent crashes and poor performance;1;Software\n"
        ).encode("utf-8")

        file = SimpleUploadedFile("semicolon.csv", csv_content, content_type="text/csv")
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Semicolon Delimiter Test",
            original_filename="semicolon.csv",
            file=file,
            status=DatasetStatus.QUEUED,
        )

        csv_info = validate_csv_structure(dataset.file.path)
        self.assertEqual(csv_info["delimiter"], ";")

        success = run_dataset_ingestion(str(dataset.id))
        self.assertTrue(success)

        dataset.refresh_from_db()
        self.assertEqual(dataset.status, DatasetStatus.COMPLETED)
        self.assertEqual(dataset.valid_rows, 2)
        reviews = list(dataset.reviews.order_by("external_review_id"))
        self.assertEqual(reviews[0].external_review_id, "SC1")
        self.assertEqual(reviews[0].product_category, "Hardware")
        self.assertEqual(reviews[0].sentiment.label, "POSITIVE")
        self.assertEqual(reviews[1].sentiment.label, "NEGATIVE")

    def test_tab_delimited_tsv_detection(self):
        """
        Verify that tab-delimited files are detected and parsed correctly.
        """
        tsv_content = (
            "review_id\treview_text\trating\tproduct_category\n"
            "TAB1\tExceptional build quality and fast shipping\t5\tElectronics\n"
        ).encode("utf-8")

        file = SimpleUploadedFile("tabs.tsv", tsv_content, content_type="text/tab-separated-values")
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Tab Delimiter Test",
            original_filename="tabs.tsv",
            file=file,
            status=DatasetStatus.QUEUED,
        )

        csv_info = validate_csv_structure(dataset.file.path)
        self.assertEqual(csv_info["delimiter"], "\t")

        success = run_dataset_ingestion(str(dataset.id))
        self.assertTrue(success)

        dataset.refresh_from_db()
        self.assertEqual(dataset.valid_rows, 1)
        rev = dataset.reviews.first()
        self.assertEqual(rev.external_review_id, "TAB1")
        self.assertEqual(rev.sentiment.label, "POSITIVE")

    def test_single_pass_inference_and_large_sub_batching(self):
        """
        Verify single-pass sentiment inference correctness and sub-batching across 2,200 items.
        """
        # Test basic correctness and class probability distribution
        sample_texts = [
            "Absolutely fantastic experience, I love this so much!",
            "It is adequate for the price, nothing special.",
            "Worst customer experience ever, completely broken product.",
            "",
            "   ",
        ]
        predictions = predict_batch(sample_texts)
        self.assertEqual(len(predictions), 5)
        self.assertEqual(predictions[0]["label"], "POSITIVE")
        self.assertEqual(predictions[1]["label"], "NEUTRAL")
        self.assertEqual(predictions[2]["label"], "NEGATIVE")
        self.assertEqual(predictions[3]["label"], "NEUTRAL")
        self.assertEqual(predictions[4]["label"], "NEUTRAL")

        # Verify probabilities sum close to 1.0
        for p in predictions[:3]:
            prob_sum = p["positive_prob"] + p["neutral_prob"] + p["negative_prob"]
            self.assertAlmostEqual(prob_sum, 1.0, delta=0.05)

        # Test sub-batch boundary handling across > 2,000 texts
        large_batch = ["Awesome product, would buy again!"] * 2200
        large_results = predict_batch(large_batch)
        self.assertEqual(len(large_results), 2200)
        self.assertEqual(large_results[0]["label"], "POSITIVE")
        self.assertEqual(large_results[2199]["label"], "POSITIVE")

    def test_streaming_csv_export_endpoint(self):
        """
        Verify that ReviewExportAPIView streams CSV output in chunks without buffering entire dataset in memory.
        """
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Export Streaming Dataset",
            original_filename="export.csv",
            status=DatasetStatus.COMPLETED,
            total_rows=5,
            valid_rows=5,
        )
        for i in range(5):
            rev = Review.objects.create(
                dataset=dataset,
                external_review_id=f"EXP_{i}",
                review_text=f"Export review text number {i}",
                rating=4.0,
                product_category="Gadgets",
            )
            SentimentResult.objects.create(
                review=rev,
                label="POSITIVE",
                confidence=0.92,
                model_version="v1.0-tfidf-logreg",
            )

        response = self.client.get(f"/api/datasets/{dataset.id}/reviews/export/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(getattr(response, "streaming", False))
        self.assertEqual(response["Content-Type"], "text/csv")
        self.assertIn("attachment; filename=", response["Content-Disposition"])

        # Stream content and parse as CSV
        streamed_bytes = b"".join(response.streaming_content)
        csv_text = streamed_bytes.decode("utf-8")
        reader = list(csv.reader(io.StringIO(csv_text)))

        # Header + 5 data rows
        self.assertEqual(len(reader), 6)
        self.assertEqual(reader[0][0], "review_id")
        self.assertEqual(reader[0][2], "review_text")
        self.assertEqual(reader[1][1], "EXP_0")
        self.assertEqual(reader[1][6], "POSITIVE")

    def test_analytics_cache_invalidation_on_state_change(self):
        """
        Verify that make_cache_key produces valid memcached-safe keys and
        clear_dataset_analytics_cache purges cached entries.
        """
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Cache Invalidation Dataset",
            original_filename="cache_test.csv",
            status=DatasetStatus.COMPLETED,
            total_rows=2,
            valid_rows=2,
        )
        Review.objects.create(
            dataset=dataset,
            review_text="Great experience",
            rating=5.0,
        )

        metrics = get_summary_metrics(dataset, use_cache=True)
        self.assertEqual(metrics["total_valid_reviews"], 1)

        key = make_cache_key("cfi_summary", str(dataset.id), dataset.completed_at)
        self.assertNotIn(" ", key)
        self.assertNotIn(":", key)
        self.assertIsNotNone(cache.get(key))

        # Invalidate cache
        clear_dataset_analytics_cache(str(dataset.id), dataset.completed_at)
        self.assertIsNone(cache.get(key))
