from django.urls import path

from . import views

app_name = "reviews"

urlpatterns = [
    # Web UI
    path("datasets/<uuid:dataset_id>/reviews/", views.review_explorer_page_view, name="explorer_page"),
    # REST API
    path("api/datasets/<uuid:dataset_id>/reviews/", views.ReviewListAPIView.as_view(), name="api_review_list"),
    path("api/datasets/<uuid:dataset_id>/reviews/<int:review_id>/", views.ReviewDetailAPIView.as_view(), name="api_review_detail"),
    path("api/datasets/<uuid:dataset_id>/reviews/export/", views.ReviewExportAPIView.as_view(), name="api_review_export"),
]
