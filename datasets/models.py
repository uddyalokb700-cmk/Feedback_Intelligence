"""
Dataset and ProcessingJob models for Customer Feedback Intelligence (CFI).
"""

import os
import uuid

from django.contrib.auth import get_user_model
from django.db import models
from django.utils.text import get_valid_filename

User = get_user_model()


def dataset_upload_path(instance, filename):
    """
    Generate an isolated, non-executable, UUID-prefixed internal storage path.
    Prevents path traversal and protects original filename.
    """
    sanitized_name = get_valid_filename(filename)
    unique_prefix = uuid.uuid4().hex[:12]
    return os.path.join("uploads", str(instance.owner_id or "system"), f"{unique_prefix}_{sanitized_name}")


class DatasetStatus(models.TextChoices):
    UPLOADED = "UPLOADED", "Uploaded"
    QUEUED = "QUEUED", "Queued"
    PROCESSING = "PROCESSING", "Processing"
    COMPLETED = "COMPLETED", "Completed"
    FAILED = "FAILED", "Failed"
    CANCELLED = "CANCELLED", "Cancelled"


class JobStatus(models.TextChoices):
    PENDING = "PENDING", "Pending"
    RUNNING = "RUNNING", "Running"
    COMPLETED = "COMPLETED", "Completed"
    FAILED = "FAILED", "Failed"
    CANCELLED = "CANCELLED", "Cancelled"


class Dataset(models.Model):
    """
    Represents an uploaded CSV dataset belonging to an authenticated user.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="datasets")
    name = models.CharField(max_length=255, help_text="User-friendly name for this dataset")
    original_filename = models.CharField(max_length=255)
    file = models.FileField(upload_to=dataset_upload_path, max_length=512)
    file_size_bytes = models.BigIntegerField(default=0)

    status = models.CharField(
        max_length=20,
        choices=DatasetStatus.choices,
        default=DatasetStatus.UPLOADED,
        db_index=True,
    )

    total_rows = models.IntegerField(default=0)
    valid_rows = models.IntegerField(default=0)
    invalid_rows = models.IntegerField(default=0)
    processed_rows = models.IntegerField(default=0)

    error_summary = models.TextField(blank=True, default="")
    validation_report = models.JSONField(default=dict, blank=True)

    model_version = models.CharField(max_length=64, default="v1.0-tfidf-logreg")
    schema_version = models.CharField(max_length=32, default="v1")

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["owner", "-created_at"], name="dataset_owner_created_idx"),
            models.Index(fields=["status"], name="dataset_status_idx"),
        ]

    def __str__(self):
        return f"{self.name} ({self.id}) - {self.status}"

    @property
    def progress_percentage(self):
        """Calculate progress percentage safely."""
        target = self.valid_rows or self.total_rows
        if not target or target == 0:
            return 100 if self.status == DatasetStatus.COMPLETED else 0
        return min(100, int((self.processed_rows / target) * 100))


class ProcessingJob(models.Model):
    """
    Observable background or synchronous processing job record for a Dataset.
    Enables progress tracking, worker diagnosis, and idempotent retry.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(
        Dataset, on_delete=models.CASCADE, related_name="processing_jobs"
    )
    status = models.CharField(
        max_length=20,
        choices=JobStatus.choices,
        default=JobStatus.PENDING,
        db_index=True,
    )

    progress_count = models.IntegerField(default=0)
    total_count = models.IntegerField(default=0)
    attempt_count = models.IntegerField(default=1)

    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    error_code = models.CharField(max_length=64, blank=True, default="")
    safe_error_message = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["dataset", "-created_at"], name="job_dataset_created_idx"),
            models.Index(fields=["status"], name="job_status_idx"),
        ]

    def __str__(self):
        return f"Job {self.id} for Dataset {self.dataset_id} [{self.status}]"
