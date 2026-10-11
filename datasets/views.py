"""
Views and API endpoints for dataset uploads, management, monitoring, and retries.
"""

import logging
import os

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, render
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from accounts.permissions import IsDatasetOwner
from datasets.models import Dataset, DatasetStatus, JobStatus, ProcessingJob
from datasets.serializers import (
    DatasetDetailSerializer,
    DatasetListSerializer,
    DatasetStatusSerializer,
    DatasetUploadSerializer,
)
from datasets.services import validate_csv_structure
from datasets.tasks import process_dataset_task

logger = logging.getLogger("datasets")


# Web Template Views
@login_required
def dashboard_view(request):
    """Render the main user dashboard with dataset overview and metrics."""
    user = request.user
    user_datasets = Dataset.objects.filter(owner=user).order_by("-created_at")
    total_datasets = user_datasets.count()
    completed_count = user_datasets.filter(status=DatasetStatus.COMPLETED).count()
    processing_count = user_datasets.filter(
        status__in=[DatasetStatus.PROCESSING, DatasetStatus.QUEUED, DatasetStatus.UPLOADED]
    ).count()
    failed_count = user_datasets.filter(status=DatasetStatus.FAILED).count()

    context = {
        "datasets": user_datasets[:10],
        "total_datasets": total_datasets,
        "completed_count": completed_count,
        "processing_count": processing_count,
        "failed_count": failed_count,
    }
    return render(request, "dashboard/index.html", context)


@login_required
def dataset_detail_page_view(request, dataset_id):
    """Render the detailed view for a single dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    return render(request, "datasets/detail.html", {"dataset": dataset})


@login_required
def dataset_upload_page_view(request):
    """Render the dataset upload page."""
    return render(request, "datasets/upload.html")


# REST API Views
class DatasetListCreateAPIView(APIView):
    """
    GET /api/datasets/ - List datasets owned by the authenticated user.
    POST /api/datasets/ - Securely upload and enqueue a CSV feedback dataset.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        datasets = Dataset.objects.filter(owner=request.user).order_by("-created_at")
        serializer = DatasetListSerializer(datasets, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = DatasetUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        uploaded_file = serializer.validated_data["file"]
        custom_name = serializer.validated_data.get("name")
        dataset_name = custom_name.strip() if custom_name else uploaded_file.name

        # Create Dataset record
        dataset = Dataset.objects.create(
            owner=request.user,
            name=dataset_name,
            original_filename=uploaded_file.name,
            file=uploaded_file,
            file_size_bytes=uploaded_file.size,
            status=DatasetStatus.QUEUED,
        )

        # Validate CSV structure immediately before queueing
        try:
            validate_csv_structure(dataset.file.path)
        except ValueError as val_err:
            # Clean up invalid file
            dataset.delete()
            return Response(
                {"error": str(val_err), "detail": "CSV validation failed."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Create observable ProcessingJob
        job = ProcessingJob.objects.create(
            dataset=dataset,
            status=JobStatus.PENDING,
        )

        # Trigger background processing
        try:
            if getattr(settings, "CELERY_TASK_ALWAYS_EAGER", True):
                # Eager/Sync dev execution
                process_dataset_task.delay(str(dataset.id), str(job.id))
            else:
                # Async Celery queue
                process_dataset_task.delay(str(dataset.id), str(job.id))
        except Exception as e:
            logger.warning("Could not dispatch via Celery broker: %s. Running synchronously.", e)
            from datasets.services import run_dataset_ingestion
            run_dataset_ingestion(str(dataset.id), str(job.id))

        # Refresh dataset from DB
        dataset.refresh_from_db()

        return Response(
            {
                "message": "Dataset uploaded and accepted for processing.",
                "dataset_id": str(dataset.id),
                "job_id": str(job.id),
                "status": dataset.status,
            },
            status=status.HTTP_202_ACCEPTED,
        )


class DatasetDetailAPIView(APIView):
    """
    GET /api/datasets/{id}/ - Retrieve detailed metadata for a dataset.
    DELETE /api/datasets/{id}/ - Securely delete dataset and all related records and files.
    """

    permission_classes = [IsAuthenticated, IsDatasetOwner]

    def get_object(self, pk, user):
        dataset = get_object_or_404(Dataset, pk=pk)
        self.check_object_permissions(self.request, dataset)
        return dataset

    def get(self, request, pk):
        dataset = self.get_object(pk, request.user)
        serializer = DatasetDetailSerializer(dataset)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, pk):
        dataset = self.get_object(pk, request.user)

        # Remove local file safely
        try:
            if dataset.file and os.path.exists(dataset.file.path):
                os.remove(dataset.file.path)
        except Exception as e:
            logger.warning("Failed to remove file from disk: %s", e)

        dataset.delete()
        return Response(
            {"message": "Dataset and all associated records permanently deleted."},
            status=status.HTTP_200_OK,
        )


class DatasetStatusAPIView(APIView):
    """
    GET /api/datasets/{id}/status/
    Fast polling endpoint for dataset progress and status.
    """

    permission_classes = [IsAuthenticated, IsDatasetOwner]

    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        self.check_object_permissions(request, dataset)
        serializer = DatasetStatusSerializer(dataset)
        return Response(serializer.data, status=status.HTTP_200_OK)


class DatasetRetryAPIView(APIView):
    """
    POST /api/datasets/{id}/retry/
    Safely retry a failed or cancelled processing job with idempotent re-ingestion.
    """

    permission_classes = [IsAuthenticated, IsDatasetOwner]

    def post(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        self.check_object_permissions(request, dataset)

        if dataset.status == DatasetStatus.PROCESSING:
            return Response(
                {"error": "Dataset is already actively processing."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Reset dataset status
        dataset.status = DatasetStatus.QUEUED
        dataset.error_summary = ""
        dataset.save(update_fields=["status", "error_summary"])

        # Create new job attempt
        prev_attempts = dataset.processing_jobs.count()
        job = ProcessingJob.objects.create(
            dataset=dataset,
            status=JobStatus.PENDING,
            attempt_count=prev_attempts + 1,
        )

        try:
            process_dataset_task.delay(str(dataset.id), str(job.id))
        except Exception:
            from datasets.services import run_dataset_ingestion
            run_dataset_ingestion(str(dataset.id), str(job.id))

        dataset.refresh_from_db()

        return Response(
            {
                "message": "Dataset processing restarted.",
                "dataset_id": str(dataset.id),
                "job_id": str(job.id),
                "status": dataset.status,
            },
            status=status.HTTP_202_ACCEPTED,
        )


class DatasetValidationReportAPIView(APIView):
    """
    GET /api/datasets/{id}/validation-report/
    Inspect row validation summary and rejected rows report.
    """

    permission_classes = [IsAuthenticated, IsDatasetOwner]

    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        self.check_object_permissions(request, dataset)
        return Response(
            {
                "dataset_id": str(dataset.id),
                "validation_report": dataset.validation_report,
                "total_rows": dataset.total_rows,
                "valid_rows": dataset.valid_rows,
                "invalid_rows": dataset.invalid_rows,
            },
            status=status.HTTP_200_OK,
        )
