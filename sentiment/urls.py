from django.urls import path

from . import views

app_name = "sentiment"

urlpatterns = [
    path("model-info/", views.model_info_page_view, name="model_info"),
    path("api/model/info/", views.ModelInfoAPIView.as_view(), name="api_model_info"),
    path("api/model/evaluation/", views.ModelEvaluationAPIView.as_view(), name="api_model_evaluation"),
]
