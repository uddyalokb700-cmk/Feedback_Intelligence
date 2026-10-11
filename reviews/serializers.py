"""
Serializers for Review records and Sentiment results.
"""

from rest_framework import serializers

from reviews.models import Review, SentimentResult


class SentimentResultSerializer(serializers.ModelSerializer):
    """Public representation of ML sentiment prediction."""

    class Meta:
        model = SentimentResult
        fields = [
            "label",
            "confidence",
            "positive_prob",
            "neutral_prob",
            "negative_prob",
            "model_version",
            "analyzed_at",
        ]


class ReviewSerializer(serializers.ModelSerializer):
    """
    Serializer for Review records with nested sentiment prediction.
    Uses select_related('sentiment') on queryset to avoid N+1 queries.
    """

    sentiment = SentimentResultSerializer(read_only=True)

    class Meta:
        model = Review
        fields = [
            "id",
            "external_review_id",
            "review_text",
            "rating",
            "review_date",
            "product_category",
            "row_number",
            "created_at",
            "sentiment",
        ]
