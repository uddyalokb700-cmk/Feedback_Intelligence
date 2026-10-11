"""
Review and SentimentResult models for Customer Feedback Intelligence (CFI).
"""

from django.db import models

from datasets.models import Dataset


class SentimentLabel(models.TextChoices):
    POSITIVE = "POSITIVE", "Positive"
    NEUTRAL = "NEUTRAL", "Neutral"
    NEGATIVE = "NEGATIVE", "Negative"


class Review(models.Model):
    """
    Normalized customer review record associated with an ingested Dataset.
    """

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="reviews")
    external_review_id = models.CharField(
        max_length=255, null=True, blank=True, db_index=True,
        help_text="Original reviewer or platform identifier from CSV"
    )
    review_text = models.TextField(help_text="Validated customer feedback text")
    rating = models.FloatField(null=True, blank=True, db_index=True, help_text="Numeric rating e.g. 1.0 to 5.0")
    review_date = models.DateField(null=True, blank=True, db_index=True, help_text="Date of review submission")
    product_category = models.CharField(max_length=128, null=True, blank=True, db_index=True)
    source_metadata = models.JSONField(default=dict, blank=True)
    row_number = models.IntegerField(null=True, blank=True, help_text="CSV file line index for provenance")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-id"]
        constraints = [
            models.UniqueConstraint(
                fields=["dataset", "external_review_id"],
                condition=models.Q(external_review_id__isnull=False) & ~models.Q(external_review_id=""),
                name="unique_external_review_per_dataset",
            )
        ]
        indexes = [
            models.Index(fields=["dataset", "review_date"], name="review_dataset_date_idx"),
            models.Index(fields=["dataset", "rating"], name="review_dataset_rating_idx"),
            models.Index(fields=["dataset", "product_category"], name="review_dataset_cat_idx"),
            models.Index(fields=["dataset", "created_at"], name="review_dataset_created_idx"),
        ]

    def __str__(self):
        return f"Review #{self.id} (Dataset {self.dataset_id})"


class SentimentResult(models.Model):
    """
    ML sentiment prediction for a review.
    Decoupled from Review to allow reproducible re-scoring and version tracking.
    """

    review = models.OneToOneField(
        Review, on_delete=models.CASCADE, related_name="sentiment"
    )
    label = models.CharField(
        max_length=16,
        choices=SentimentLabel.choices,
        db_index=True,
    )
    confidence = models.FloatField(
        null=True,
        blank=True,
        help_text="Model predicted class probability (calibrated estimate)",
    )
    positive_prob = models.FloatField(null=True, blank=True)
    neutral_prob = models.FloatField(null=True, blank=True)
    negative_prob = models.FloatField(null=True, blank=True)
    model_version = models.CharField(max_length=64, db_index=True)
    analyzed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-id"]
        indexes = [
            models.Index(fields=["label"], name="sentiment_label_idx"),
            models.Index(fields=["model_version"], name="sentiment_model_ver_idx"),
        ]

    def __str__(self):
        return f"{self.label} ({self.confidence:.2f} if self.confidence else 'N/A') [Model {self.model_version}]"
