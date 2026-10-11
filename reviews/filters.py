"""
FilterSets for Review records.
Prevents arbitrary ORM injection and enforces strict field validation.
"""

import django_filters
from django.db.models import Q

from reviews.models import Review


class ReviewFilter(django_filters.FilterSet):
    sentiment = django_filters.CharFilter(
        field_name="sentiment__label",
        lookup_expr="iexact",
        help_text="Filter by sentiment: POSITIVE, NEUTRAL, or NEGATIVE"
    )
    rating = django_filters.NumberFilter(
        field_name="rating",
        help_text="Filter by exact numeric rating"
    )
    min_rating = django_filters.NumberFilter(
        field_name="rating",
        lookup_expr="gte",
        help_text="Filter reviews with rating greater than or equal to value"
    )
    max_rating = django_filters.NumberFilter(
        field_name="rating",
        lookup_expr="lte",
        help_text="Filter reviews with rating less than or equal to value"
    )
    category = django_filters.CharFilter(
        field_name="product_category",
        lookup_expr="iexact",
        help_text="Filter by exact category"
    )
    date_from = django_filters.DateFilter(
        field_name="review_date",
        lookup_expr="gte",
        help_text="Filter reviews submitted on or after date (YYYY-MM-DD)"
    )
    date_to = django_filters.DateFilter(
        field_name="review_date",
        lookup_expr="lte",
        help_text="Filter reviews submitted on or before date (YYYY-MM-DD)"
    )
    search = django_filters.CharFilter(
        method="filter_by_search",
        help_text="Search text in review body or review ID"
    )

    class Meta:
        model = Review
        fields = [
            "sentiment",
            "rating",
            "min_rating",
            "max_rating",
            "category",
            "date_from",
            "date_to",
            "search",
        ]

    def filter_by_search(self, queryset, name, value):
        if not value or not value.strip():
            return queryset
        val = value.strip()
        return queryset.filter(
            Q(review_text__icontains=val) | Q(external_review_id__icontains=val)
        )
