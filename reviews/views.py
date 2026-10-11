"""
Views and API endpoints for review listing, filtering, detail inspection, and export.
"""

import csv

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from accounts.permissions import IsDatasetOwner
from datasets.models import Dataset
from reviews.filters import ReviewFilter
from reviews.models import Review
from reviews.serializers import ReviewSerializer


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# Web Template Views
@login_required
def review_explorer_page_view(request, dataset_id):
    """Render the interactive Review Explorer page."""
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    # Distinct categories for filter dropdown
    categories = (
        Review.objects.filter(dataset=dataset, product_category__isnull=False)
        .exclude(product_category="")
        .values_list("product_category", flat=True)
        .distinct()
        .order_by("product_category")
    )
    return render(
        request,
        "reviews/explorer.html",
        {"dataset": dataset, "categories": list(categories)},
    )


# REST API Views
class ReviewListAPIView(APIView):
    """
    GET /api/datasets/{id}/reviews/
    Retrieve paginated, filtered, and sorted reviews with predicted sentiment.
    Zero N+1 queries via select_related('sentiment').
    """

    permission_classes = [IsAuthenticated, IsDatasetOwner]
    pagination_class = StandardPagination

    def get_dataset(self, dataset_id, user):
        dataset = get_object_or_404(Dataset, id=dataset_id)
        self.check_object_permissions(self.request, dataset)
        return dataset

    def get(self, request, dataset_id):
        dataset = self.get_dataset(dataset_id, request.user)

        queryset = (
            Review.objects.filter(dataset=dataset)
            .select_related("sentiment")
            .order_by("-id")
        )

        # Apply filtering
        filterset = ReviewFilter(request.GET, queryset=queryset)
        if not filterset.is_valid():
            return Response(filterset.errors, status=status.HTTP_400_BAD_REQUEST)
        filtered_qs = filterset.qs

        # Apply supported ordering
        ordering = request.GET.get("ordering")
        allowed_ordering = {
            "id": "id",
            "-id": "-id",
            "rating": "rating",
            "-rating": "-rating",
            "date": "review_date",
            "-date": "-review_date",
            "review_date": "review_date",
            "-review_date": "-review_date",
            "confidence": "sentiment__confidence",
            "-confidence": "-sentiment__confidence",
        }
        if ordering and ordering in allowed_ordering:
            filtered_qs = filtered_qs.order_by(allowed_ordering[ordering])

        paginator = self.pagination_class()
        paginated_reviews = paginator.paginate_queryset(filtered_qs, request)
        serializer = ReviewSerializer(paginated_reviews, many=True)
        return paginator.get_paginated_response(serializer.data)


class ReviewDetailAPIView(APIView):
    """
    GET /api/datasets/{id}/reviews/{review_id}/
    Retrieve a single review record with full sentiment breakdown.
    """

    permission_classes = [IsAuthenticated, IsDatasetOwner]

    def get(self, request, dataset_id, review_id):
        dataset = get_object_or_404(Dataset, id=dataset_id)
        self.check_object_permissions(request, dataset)

        review = get_object_or_404(
            Review.objects.select_related("sentiment"),
            dataset=dataset,
            id=review_id,
        )
        serializer = ReviewSerializer(review)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ReviewExportAPIView(APIView):
    """
    GET /api/datasets/{id}/reviews/export/
    Export authorized dataset reviews to a clean CSV file with applied filters.
    """

    permission_classes = [IsAuthenticated, IsDatasetOwner]

    def get(self, request, dataset_id):
        dataset = get_object_or_404(Dataset, id=dataset_id)
        self.check_object_permissions(request, dataset)

        queryset = (
            Review.objects.filter(dataset=dataset)
            .select_related("sentiment")
            .order_by("id")
        )

        filterset = ReviewFilter(request.GET, queryset=queryset)
        if filterset.is_valid():
            queryset = filterset.qs

        response = HttpResponse(content_type="text/csv")
        safe_name = dataset.name.replace(" ", "_").lower()
        response["Content-Disposition"] = f'attachment; filename="{safe_name}_reviews_export.csv"'

        writer = csv.writer(response)
        writer.writerow([
            "review_id",
            "external_review_id",
            "review_text",
            "rating",
            "review_date",
            "product_category",
            "sentiment_label",
            "sentiment_confidence",
            "model_version",
        ])

        # Bounded streaming iterator for memory efficiency
        for review in queryset.iterator(chunk_size=1000):
            sent = getattr(review, "sentiment", None)
            writer.writerow([
                review.id,
                review.external_review_id or "",
                review.review_text,
                review.rating if review.rating is not None else "",
                review.review_date.isoformat() if review.review_date else "",
                review.product_category or "",
                sent.label if sent else "",
                sent.confidence if sent and sent.confidence is not None else "",
                sent.model_version if sent else "",
            ])

        return response
