"""
Serializers for Dataset management and upload validation.
"""

from django.conf import settings
from rest_framework import serializers

from datasets.models import Dataset, ProcessingJob


class ProcessingJobSerializer(serializers.ModelSerializer):
    """Serializer for background processing job monitoring."""

    class Meta:
        model = ProcessingJob
        fields = [
            "id",
            "dataset_id",
            "status",
            "progress_count",
            "total_count",
            "attempt_count",
            "started_at",
            "completed_at",
            "error_code",
            "safe_error_message",
            "created_at",
        ]
        read_only_fields = fields


class DatasetListSerializer(serializers.ModelSerializer):
    """Compact serializer for dashboard listing."""

    progress_percentage = serializers.ReadOnlyField()

    class Meta:
        model = Dataset
        fields = [
            "id",
            "name",
            "original_filename",
            "file_size_bytes",
            "status",
            "total_rows",
            "valid_rows",
            "invalid_rows",
            "processed_rows",
            "progress_percentage",
            "created_at",
            "completed_at",
        ]
        read_only_fields = fields


class DatasetDetailSerializer(serializers.ModelSerializer):
    """Detailed dataset metadata including validation reports."""

    progress_percentage = serializers.ReadOnlyField()
    latest_job = serializers.SerializerMethodField()

    class Meta:
        model = Dataset
        fields = [
            "id",
            "name",
            "original_filename",
            "file_size_bytes",
            "status",
            "total_rows",
            "valid_rows",
            "invalid_rows",
            "processed_rows",
            "progress_percentage",
            "error_summary",
            "validation_report",
            "model_version",
            "schema_version",
            "created_at",
            "started_at",
            "completed_at",
            "latest_job",
        ]
        read_only_fields = fields

    def get_latest_job(self, obj):
        latest = obj.processing_jobs.first()
        if latest:
            return ProcessingJobSerializer(latest).data
        return None


class DatasetStatusSerializer(serializers.ModelSerializer):
    """Fast polling serializer for dataset status checks."""

    progress_percentage = serializers.ReadOnlyField()

    class Meta:
        model = Dataset
        fields = [
            "id",
            "status",
            "total_rows",
            "valid_rows",
            "processed_rows",
            "progress_percentage",
            "error_summary",
            "started_at",
            "completed_at",
        ]
        read_only_fields = fields


class DatasetUploadSerializer(serializers.Serializer):
    """Serializer for CSV upload and initial schema validation."""

    file = serializers.FileField(required=True)
    name = serializers.CharField(required=False, max_length=255, allow_blank=True)

    def validate_file(self, value):
        # 1. Filename validation
        filename = value.name.lower()
        if not filename.endswith(".csv"):
            raise serializers.ValidationError("Only .csv files are supported.")

        # 2. File size check
        max_size = getattr(settings, "MAX_UPLOAD_SIZE_BYTES", 25 * 1024 * 1024)
        if value.size > max_size:
            max_mb = max_size / (1024 * 1024)
            raise serializers.ValidationError(
                f"File exceeds maximum upload size of {max_mb:.0f} MB."
            )

        if value.size == 0:
            raise serializers.ValidationError("Uploaded file is empty (0 bytes).")

        return value
