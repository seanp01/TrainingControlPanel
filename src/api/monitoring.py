"""
Monitoring API endpoints for tracking training progress and system health.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from src.monitoring.metrics_collector import MetricsCollector
from src.monitoring.drift_detector import DriftDetector


router = APIRouter()


class MetricsResponse(BaseModel):
    """Response model for metrics data."""
    timestamp: str
    system: Dict[str, Any]
    training: Dict[str, Any]


class DriftResponse(BaseModel):
    """Response model for drift detection data."""
    model_name: str
    drift_score: float
    threshold: float
    alert_triggered: bool
    timestamp: str


class TrainingJobStatus(BaseModel):
    """Response model for training job status."""
    job_id: str
    name: str
    status: str
    progress: float
    started_at: Optional[str]
    estimated_completion: Optional[str]
    metrics: Dict[str, Any]


def get_metrics_collector() -> MetricsCollector:
    """Dependency to get metrics collector instance."""
    # This would typically be injected from the app state
    from src.main import app
    return app.state.metrics_collector


def get_drift_detector() -> DriftDetector:
    """Dependency to get drift detector instance."""
    return DriftDetector()


@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring service."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.get("/metrics", response_model=MetricsResponse)
async def get_current_metrics(
    metrics_collector: MetricsCollector = Depends(get_metrics_collector)
):
    """Get current system and training metrics."""
    try:
        metrics = metrics_collector.get_current_metrics()
        if not metrics:
            raise HTTPException(status_code=500, detail="Failed to collect metrics")
        
        return MetricsResponse(**metrics)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/history")
async def get_metrics_history(
    hours: int = Query(default=24, ge=1, le=168),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector)
):
    """Get metrics history for the specified number of hours."""
    try:
        history = metrics_collector.get_metrics_history(hours=hours)
        return {"history": history, "hours": hours}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/prometheus")
async def get_prometheus_metrics(
    metrics_collector: MetricsCollector = Depends(get_metrics_collector)
):
    """Get metrics in Prometheus format."""
    try:
        prometheus_data = metrics_collector.get_prometheus_metrics()
        return {"metrics": prometheus_data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/summary")
async def get_training_summary(
    metrics_collector: MetricsCollector = Depends(get_metrics_collector)
):
    """Get training activity summary."""
    try:
        summary = metrics_collector.get_training_summary()
        return summary
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/jobs")
async def get_training_jobs():
    """Get list of all training jobs with their current status."""
    try:
        # This would typically query the scheduler for job status
        jobs = [
            TrainingJobStatus(
                job_id="job_1",
                name="Main Model Training",
                status="running",
                progress=0.65,
                started_at="2025-06-29T10:00:00Z",
                estimated_completion="2025-06-29T14:30:00Z",
                metrics={"loss": 0.45, "accuracy": 0.82}
            ),
            TrainingJobStatus(
                job_id="job_2",
                name="Assistant Model Fine-tuning",
                status="scheduled",
                progress=0.0,
                started_at=None,
                estimated_completion="2025-06-29T16:00:00Z",
                metrics={}
            )
        ]
        
        return {"jobs": jobs}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/jobs/{job_id}")
async def get_training_job_details(job_id: str):
    """Get detailed information about a specific training job."""
    try:
        # This would typically query the database for job details
        job_details = {
            "job_id": job_id,
            "name": "Main Model Training",
            "status": "running",
            "progress": 0.65,
            "started_at": "2025-06-29T10:00:00Z",
            "estimated_completion": "2025-06-29T14:30:00Z",
            "config": {
                "model_type": "transformer",
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10
            },
            "metrics": {
                "current_epoch": 6,
                "current_loss": 0.45,
                "current_accuracy": 0.82,
                "best_accuracy": 0.85,
                "training_time": "4h 30m"
            },
            "logs": [
                {"timestamp": "2025-06-29T10:00:00Z", "level": "INFO", "message": "Training started"},
                {"timestamp": "2025-06-29T11:30:00Z", "level": "INFO", "message": "Epoch 3 completed"},
                {"timestamp": "2025-06-29T13:00:00Z", "level": "INFO", "message": "Checkpoint saved"}
            ]
        }
        
        return job_details
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift/summary")
async def get_drift_summary(
    drift_detector: DriftDetector = Depends(get_drift_detector)
):
    """Get drift detection summary across all models."""
    try:
        summary = drift_detector.get_drift_summary()
        return summary
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift/{model_name}")
async def get_model_drift(
    model_name: str,
    hours: int = Query(default=168, ge=1, le=720),
    drift_detector: DriftDetector = Depends(get_drift_detector)
):
    """Get drift history for a specific model."""
    try:
        history = drift_detector.get_drift_history(model_name, hours=hours)
        return {
            "model_name": model_name,
            "drift_history": history,
            "hours": hours
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift/{model_name}/test")
async def run_drift_tests(
    model_name: str,
    drift_detector: DriftDetector = Depends(get_drift_detector)
):
    """Run comprehensive drift tests for a specific model."""
    try:
        test_results = await drift_detector.run_drift_tests(model_name)
        return test_results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_active_alerts():
    """Get list of active monitoring alerts."""
    try:
        # This would typically query an alerts database
        alerts = [
            {
                "id": "alert_1",
                "type": "drift",
                "severity": "high",
                "model_name": "main_model",
                "message": "High drift detected (score: 0.15)",
                "timestamp": "2025-06-29T12:30:00Z",
                "status": "active"
            },
            {
                "id": "alert_2",
                "type": "system",
                "severity": "medium",
                "message": "High memory usage (85%)",
                "timestamp": "2025-06-29T13:15:00Z",
                "status": "active"
            }
        ]
        
        return {"alerts": alerts}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_dashboard_data(
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
    drift_detector: DriftDetector = Depends(get_drift_detector)
):
    """Get comprehensive dashboard data."""
    try:
        # Collect data from various sources
        current_metrics = metrics_collector.get_current_metrics()
        training_summary = metrics_collector.get_training_summary()
        drift_summary = drift_detector.get_drift_summary()
        
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_metrics": current_metrics.get("system", {}),
            "training_summary": training_summary,
            "drift_summary": drift_summary,
            "recent_metrics": metrics_collector.get_metrics_history(hours=6),
            "status": "healthy"  # Would be calculated based on various factors
        }
        
        return dashboard_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
