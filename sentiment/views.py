"""
Views and API endpoints for model inspection and evaluation metrics.
"""

from django.shortcuts import render
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from sentiment.inference import get_model_metadata


def model_info_page_view(request):
    """Render the model information and evaluation page."""
    metadata = get_model_metadata()
    return render(request, "model/info.html", {"metadata": metadata})


class ModelInfoAPIView(APIView):
    """
    GET /api/model/info/
    Retrieve information about the active sentiment model architecture and limitations.
    """

    permission_classes = [AllowAny]

    def get(self, request):
        metadata = get_model_metadata()
        info = {
            "model_version": metadata.get("model_version"),
            "model_type": metadata.get("model_type"),
            "classes": metadata.get("classes", []),
            "hyperparameters": metadata.get("hyperparameters", {}),
            "known_limitations": metadata.get("known_limitations", []),
            "trained_at": metadata.get("trained_at"),
            "sample_count": metadata.get("sample_count"),
        }
        return Response(info, status=status.HTTP_200_OK)


class ModelEvaluationAPIView(APIView):
    """
    GET /api/model/evaluation/
    Retrieve detailed evaluation metrics and confusion matrix from held-out validation.
    """

    permission_classes = [AllowAny]

    def get(self, request):
        metadata = get_model_metadata()
        eval_data = {
            "model_version": metadata.get("model_version"),
            "evaluation_metrics": metadata.get("evaluation_metrics"),
            "confusion_matrix": metadata.get("confusion_matrix"),
            "train_size": metadata.get("train_size"),
            "test_size": metadata.get("test_size"),
        }
        return Response(eval_data, status=status.HTTP_200_OK)
