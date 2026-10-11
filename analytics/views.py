"""
Views and REST API endpoints for dataset analytics, trends, and business intelligence.
"""

from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, render
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from accounts.permissions import IsDatasetOwner
from analytics.services import (
    get_category_metrics,
    get_sentiment_distribution,
    get_summary_metrics,
    get_top_terms,
    get_trends,
)
from datasets.models import Dataset


# Web Template Views
@login_required
def analytics_dashboard_page_view(request, dataset_id):
    """Render the comprehensive interactive Analytics Dashboard page."""
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    summary = get_summary_metrics(dataset)
    return render(
        request,
        "dashboard/analytics.html",
        {"dataset": dataset, "summary": summary},
    )


# REST API Views
class BaseDatasetAnalyticsView(APIView):
    """Base class providing dataset resolution and strict ownership authorization."""

    permission_classes = [IsAuthenticated, IsDatasetOwner]

    def get_dataset(self, dataset_id, request):
        dataset = get_object_or_404(Dataset, id=dataset_id)
        self.check_object_permissions(request, dataset)
        return dataset


class SummaryAPIView(BaseDatasetAnalyticsView):
    """
    GET /api/datasets/{id}/summary/
    Retrieve summary KPIs: counts, proportions, average rating, negative review rate.
    """

    def get(self, request, dataset_id):
        dataset = self.get_dataset(dataset_id, request)
        summary = get_summary_metrics(dataset)
        return Response(summary, status=status.HTTP_200_OK)


class SentimentDistributionAPIView(BaseDatasetAnalyticsView):
    """
    GET /api/datasets/{id}/sentiment-distribution/
    Retrieve breakdown of positive, neutral, and negative sentiment counts and proportions.
    """

    def get(self, request, dataset_id):
        dataset = self.get_dataset(dataset_id, request)
        dist = get_sentiment_distribution(dataset)
        return Response(dist, status=status.HTTP_200_OK)


class TrendsAPIView(BaseDatasetAnalyticsView):
    """
    GET /api/datasets/{id}/trends/
    Retrieve time-series trends: review volume, average rating, and sentiment over time.
    """

    def get(self, request, dataset_id):
        dataset = self.get_dataset(dataset_id, request)
        trends = get_trends(dataset)
        return Response(trends, status=status.HTTP_200_OK)


class CategoriesAPIView(BaseDatasetAnalyticsView):
    """
    GET /api/datasets/{id}/categories/
    Retrieve volume, average rating, and sentiment metrics grouped by product category.
    """

    def get(self, request, dataset_id):
        dataset = self.get_dataset(dataset_id, request)
        categories = get_category_metrics(dataset)
        return Response(categories, status=status.HTTP_200_OK)


class TopTermsAPIView(BaseDatasetAnalyticsView):
    """
    GET /api/datasets/{id}/top-terms/
    Retrieve top recurring keywords in positive and negative reviews.
    """

    def get(self, request, dataset_id):
        dataset = self.get_dataset(dataset_id, request)
        limit = min(int(request.GET.get("limit", 15)), 50)
        terms = get_top_terms(dataset, limit=limit)
        return Response(terms, status=status.HTTP_200_OK)
