"""
Object-level authorization permissions for Customer Feedback Intelligence.
Strictly prevents horizontal privilege escalation between users.
"""

from rest_framework import permissions


class IsDatasetOwner(permissions.BasePermission):
    """
    Object-level permission allowing only the owner of a dataset to view or modify it.
    Works for Dataset, Review, ProcessingJob, and related analytics endpoints.
    """

    def has_object_permission(self, request, view, obj):
        if not request.user or not request.user.is_authenticated:
            return False

        # If obj has an 'owner' attribute directly (e.g. Dataset)
        if hasattr(obj, "owner"):
            return obj.owner == request.user

        # If obj has a 'dataset' attribute (e.g. Review, ProcessingJob)
        if hasattr(obj, "dataset"):
            return obj.dataset.owner == request.user

        return False
