"""
Data ingestion, CSV validation, and bounded chunk processing service.
"""

import csv
import logging
import os
from datetime import date

from dateutil import parser as date_parser
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from datasets.models import Dataset, DatasetStatus, JobStatus, ProcessingJob
from reviews.models import Review, SentimentResult
from sentiment.inference import predict_batch

logger = logging.getLogger("datasets")

# Canonical column name aliases
HEADER_SYNONYMS = {
    "review_text": [
        "review_text", "review", "reviews", "feedback", "comment", "comments",
        "text", "message", "body", "content", "customer_review", "review_body"
    ],
    "external_review_id": [
        "review_id", "id", "external_id", "feedback_id", "external_review_id",
        "review_number", "order_id"
    ],
    "rating": [
        "rating", "ratings", "score", "scores", "stars", "star_rating",
        "user_rating", "review_rating"
    ],
    "review_date": [
        "review_date", "date", "created_at", "timestamp", "time", "submission_date",
        "posted_date", "review_time"
    ],
    "product_category": [
        "product_category", "category", "categories", "dept", "department",
        "product", "item_category", "product_type", "product_name"
    ],
}


def map_headers(fieldnames: list[str]) -> dict:
    """
    Map CSV header strings to canonical CFI fields.
    Case-insensitive, whitespace-stripped matching.
    """
    mapping = {}
    if not fieldnames:
        return mapping

    normalized = {f.strip().lower().replace(" ", "_"): f for f in fieldnames if f}

    for canonical, synonyms in HEADER_SYNONYMS.items():
        for syn in synonyms:
            if syn in normalized:
                mapping[canonical] = normalized[syn]
                break

    return mapping


def validate_csv_structure(file_path: str) -> dict:
    """
    Inspect the CSV headers and validate basic structure.
    Raises ValueError with a user-friendly message if invalid.
    """
    if not os.path.exists(file_path):
        raise ValueError("Uploaded file could not be found on server.")

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError("The uploaded CSV file is empty (0 bytes).")

    max_size = getattr(settings, "MAX_UPLOAD_SIZE_BYTES", 25 * 1024 * 1024)
    if file_size > max_size:
        raise ValueError(f"File size ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed limit of {max_size / (1024*1024):.0f} MB.")

    # Detect encoding and read header
    try:
        with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.reader(f)
            header_row = next(reader, None)
    except Exception as e:
        raise ValueError(f"Could not read CSV file: {e!s}") from e

    if not header_row:
        raise ValueError("CSV file has no header row or content.")

    mapping = map_headers(header_row)

    if "review_text" not in mapping:
        supported = ", ".join(HEADER_SYNONYMS["review_text"][:6])
        raise ValueError(
            f"Missing required review text column. Found columns: {', '.join(header_row)}. "
            f"Please include a column named one of: {supported}."
        )

    return {
        "header_row": header_row,
        "mapping": mapping,
        "file_size": file_size,
    }


def parse_rating(val) -> float | None:
    """Validate and parse numeric rating (e.g. 1.0 - 5.0 or 1 - 10)."""
    if val is None or val == "":
        return None
    try:
        f = float(str(val).strip())
        if 0.0 <= f <= 10.0:
            return round(f, 2)
        return None
    except (ValueError, TypeError):
        return None


def parse_date(val) -> date | None:
    """Safely parse various date formats into standard datetime.date."""
    if not val or not str(val).strip():
        return None
    try:
        dt = date_parser.parse(str(val).strip())
        return dt.date()
    except Exception:
        return None


def run_dataset_ingestion(dataset_id: str, job_id: str = None) -> bool:
    """
    Execute full bounded chunk ingestion and sentiment classification for a dataset.
    Idempotent and transactionally bounded.
    """
    dataset = Dataset.objects.get(id=dataset_id)
    job = None
    if job_id:
        job = ProcessingJob.objects.filter(id=job_id).first()

    logger.info("Starting processing for Dataset %s (Job %s)", dataset_id, job_id)

    # Update states
    now = timezone.now()
    dataset.status = DatasetStatus.PROCESSING
    dataset.started_at = now
    dataset.save(update_fields=["status", "started_at"])

    if job:
        job.status = JobStatus.RUNNING
        job.started_at = now
        job.save(update_fields=["status", "started_at"])

    try:
        file_path = dataset.file.path
        csv_info = validate_csv_structure(file_path)
        mapping = csv_info["mapping"]

        # Clear existing reviews if this is a retry to prevent duplicates
        with transaction.atomic():
            dataset.reviews.all().delete()
            dataset.processed_rows = 0
            dataset.valid_rows = 0
            dataset.invalid_rows = 0
            dataset.save(update_fields=["processed_rows", "valid_rows", "invalid_rows"])

        text_col = mapping["review_text"]
        id_col = mapping.get("external_review_id")
        rating_col = mapping.get("rating")
        date_col = mapping.get("review_date")
        cat_col = mapping.get("product_category")

        batch_size = getattr(settings, "INGESTION_BATCH_SIZE", 1000)
        seen_external_ids = set()

        total_rows_count = 0
        valid_rows_count = 0
        invalid_rows_count = 0
        rejection_samples = []

        batch_reviews = []

        with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)

            for line_idx, row in enumerate(reader, start=2):
                total_rows_count += 1

                raw_text = row.get(text_col, "")
                if not raw_text or not raw_text.strip():
                    invalid_rows_count += 1
                    if len(rejection_samples) < 25:
                        rejection_samples.append({
                            "row": line_idx,
                            "reason": "Missing or empty review text",
                            "sample": "",
                        })
                    continue

                text = raw_text.strip()
                ext_id = row.get(id_col, "").strip() if id_col else ""
                if ext_id and ext_id in seen_external_ids:
                    invalid_rows_count += 1
                    if len(rejection_samples) < 25:
                        rejection_samples.append({
                            "row": line_idx,
                            "reason": f"Duplicate review_id: '{ext_id}'",
                            "sample": text[:50],
                        })
                    continue

                if ext_id:
                    seen_external_ids.add(ext_id)

                rating = parse_rating(row.get(rating_col)) if rating_col else None
                review_date = parse_date(row.get(date_col)) if date_col else None
                category = row.get(cat_col, "").strip() if cat_col else ""

                valid_rows_count += 1
                review_obj = Review(
                    dataset=dataset,
                    external_review_id=ext_id or None,
                    review_text=text,
                    rating=rating,
                    review_date=review_date,
                    product_category=category[:128] if category else None,
                    row_number=line_idx,
                )
                batch_reviews.append(review_obj)

                # Process batch when threshold is reached
                if len(batch_reviews) >= batch_size:
                    _process_review_batch(dataset, job, batch_reviews)
                    batch_reviews = []

            # Flush remaining batch
            if batch_reviews:
                _process_review_batch(dataset, job, batch_reviews)
                batch_reviews = []

        # Finalize dataset metadata and completion
        completion_time = timezone.now()
        dataset.total_rows = total_rows_count
        dataset.valid_rows = valid_rows_count
        dataset.invalid_rows = invalid_rows_count
        dataset.processed_rows = valid_rows_count
        dataset.status = DatasetStatus.COMPLETED
        dataset.completed_at = completion_time
        dataset.validation_report = {
            "mapped_columns": mapping,
            "total_rows": total_rows_count,
            "valid_rows": valid_rows_count,
            "invalid_rows": invalid_rows_count,
            "rejections": rejection_samples,
        }
        dataset.save()

        if job:
            job.status = JobStatus.COMPLETED
            job.progress_count = valid_rows_count
            job.total_count = valid_rows_count
            job.completed_at = completion_time
            job.save()

        logger.info(
            "Successfully completed processing Dataset %s: %d valid reviews, %d invalid.",
            dataset_id, valid_rows_count, invalid_rows_count
        )
        return True

    except Exception as exc:
        logger.exception("Processing failed for Dataset %s: %s", dataset_id, exc)
        safe_msg = str(exc)
        dataset.status = DatasetStatus.FAILED
        dataset.error_summary = safe_msg
        dataset.save(update_fields=["status", "error_summary"])

        if job:
            job.status = JobStatus.FAILED
            job.error_code = exc.__class__.__name__
            job.safe_error_message = safe_msg
            job.save(update_fields=["status", "error_code", "safe_error_message"])
        return False


def _process_review_batch(dataset: Dataset, job: ProcessingJob | None, reviews: list[Review]):
    """
    Atomic bulk write and batched inference for a bounded chunk of reviews.
    """
    if not reviews:
        return

    with transaction.atomic():
        # 1. Bulk create reviews
        created_reviews = Review.objects.bulk_create(reviews, batch_size=len(reviews))

        # 2. Batch sentiment prediction
        texts = [r.review_text for r in created_reviews]
        predictions = predict_batch(texts, model_version=dataset.model_version)

        # 3. Create SentimentResult objects
        sentiment_records = []
        for review_obj, pred in zip(created_reviews, predictions):
            sentiment_records.append(
                SentimentResult(
                    review=review_obj,
                    label=pred["label"],
                    confidence=pred["confidence"],
                    positive_prob=pred["positive_prob"],
                    neutral_prob=pred["neutral_prob"],
                    negative_prob=pred["negative_prob"],
                    model_version=pred["model_version"],
                )
            )

        SentimentResult.objects.bulk_create(sentiment_records, batch_size=len(sentiment_records))

        # 4. Update progress counters
        dataset.processed_rows = Review.objects.filter(dataset=dataset).count()
        dataset.save(update_fields=["processed_rows"])

        if job:
            job.progress_count = dataset.processed_rows
            job.save(update_fields=["progress_count"])
