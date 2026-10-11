from django.urls import path

from . import views

app_name = "analytics"

urlpatterns = [
    # Web UI
    path("datasets/<uuid:dataset_id>/analytics/", views.analytics_dashboard_page_view, name="analytics_dashboard"),
    # REST API
    path("api/datasets/<uuid:dataset_id>/summary/", views.SummaryAPIView.as_view(), name="api_summary"),
    path("api/datasets/<uuid:dataset_id>/sentiment-distribution/", views.SentimentDistributionAPIView.as_view(), name="api_sentiment_distribution"),
    path("api/datasets/<uuid:dataset_id>/trends/", views.TrendsAPIView.as_view(), name="api_trends"),
    path("api/datasets/<uuid:dataset_id>/categories/", views.CategoriesAPIView.as_view(), name="api_categories"),
    path("api/datasets/<uuid:dataset_id>/top-terms/", views.TopTermsAPIView.as_view(), name="api_top_terms"),
]
