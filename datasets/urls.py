from django.urls import path

from . import views

app_name = "datasets"

urlpatterns = [
    # Web UI views
    path("", views.dashboard_view, name="index"),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("datasets/upload/", views.dataset_upload_page_view, name="upload_page"),
    path("datasets/<uuid:dataset_id>/", views.dataset_detail_page_view, name="detail_page"),
    # REST API routes
    path("api/datasets/", views.DatasetListCreateAPIView.as_view(), name="api_dataset_list_create"),
    path("api/datasets/<uuid:pk>/", views.DatasetDetailAPIView.as_view(), name="api_dataset_detail"),
    path("api/datasets/<uuid:pk>/status/", views.DatasetStatusAPIView.as_view(), name="api_dataset_status"),
    path("api/datasets/<uuid:pk>/retry/", views.DatasetRetryAPIView.as_view(), name="api_dataset_retry"),
    path("api/datasets/<uuid:pk>/validation-report/", views.DatasetValidationReportAPIView.as_view(), name="api_dataset_validation_report"),
]
