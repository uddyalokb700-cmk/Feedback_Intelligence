"""
Celery background tasks for dataset ingestion and processing.
"""

import logging

from celery import shared_task

from datasets.services import run_dataset_ingestion

logger = logging.getLogger("datasets")


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_dataset_task(self, dataset_id: str, job_id: str = None):
    """
    Celery task to asynchronously process a dataset in bounded batches.
    """
    logger.info("Executing process_dataset_task for dataset=%s, job=%s", dataset_id, job_id)
    try:
        success = run_dataset_ingestion(dataset_id=dataset_id, job_id=job_id)
        return {"dataset_id": dataset_id, "success": success}
    except Exception as exc:
        logger.exception("Error executing process_dataset_task: %s", exc)
        raise exc
