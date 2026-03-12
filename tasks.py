# tasks.py — Celery background tasks for HydraGIS
#
# Start worker:  celery -A tasks worker --loglevel=info
# ──────────────────────────────────────────────────────────────────────────────

import logging
from celery import Celery
from config import settings

log = logging.getLogger("hydragis.tasks")

# ── Celery app ────────────────────────────────────────────────────────────────
celery_app = Celery(
    "hydragis",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    # Keep results for 24 h so /model/train/status can be polled
    result_expires=86400,
)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING TASK
# ══════════════════════════════════════════════════════════════════════════════

@celery_app.task(bind=True, name="tasks.train_models_task")
def train_models_task(self):
    """
    Runs train_all() in the background.
    Returns the TrainMetrics dict on success.

    Poll status via:
        GET /model/train/status/{task_id}
    """
    try:
        self.update_state(state="PROGRESS", meta={"step": "loading data"})
        log.info("Background training started (task_id=%s)", self.request.id)

        from models.train import train_all
        self.update_state(state="PROGRESS", meta={"step": "training RF + XGBoost"})
        metrics = train_all()

        # Invalidate the Redis ward-score cache so next request recomputes fresh
        try:
            import redis as redis_lib
            r = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
            r.delete("hydragis:ward_scores")
            log.info("Redis cache invalidated after background training")
        except Exception as cache_err:
            log.warning("Could not invalidate Redis cache: %s", cache_err)

        log.info("Background training complete — RF r²=%.3f  XGB r²=%.3f",
                 metrics["random_forest"]["r2"], metrics["xgboost"]["r2"])
        return metrics

    except Exception as exc:
        log.exception("Training task failed: %s", exc)
        raise self.retry(exc=exc, max_retries=0)   # don't retry; surface failure
