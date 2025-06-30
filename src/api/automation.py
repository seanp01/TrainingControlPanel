"""
Automation API endpoints for managing training schedules and workflows.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from src.automation.scheduler import TrainingScheduler, TrainingJob


router = APIRouter()


class CreateJobRequest(BaseModel):
    """Request model for creating a training job."""
    name: str
    pipeline_config: Dict[str, Any]
    schedule: str
    enabled: bool = True


class JobResponse(BaseModel):
    """Response model for training job information."""
    job_id: str
    name: str
    schedule: str
    enabled: bool
    status: str
    created_at: str
    last_run: Optional[str]
    next_run: Optional[str]


class TriggerJobResponse(BaseModel):
    """Response model for job trigger operation."""
    execution_id: str
    message: str


def get_scheduler() -> TrainingScheduler:
    """Dependency to get scheduler instance."""
    from src.main import app
    return app.state.scheduler


@router.get("/health")
async def health_check():
    """Health check endpoint for automation service."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.post("/jobs", response_model=JobResponse)
async def create_training_job(
    request: CreateJobRequest,
    scheduler: TrainingScheduler = Depends(get_scheduler)
):
    """Create a new automated training job."""
    try:
        job_id = await scheduler.add_job(
            name=request.name,
            pipeline_config=request.pipeline_config,
            schedule=request.schedule
        )
        
        job = scheduler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=500, detail="Failed to create job")
        
        return JobResponse(
            job_id=job.job_id,
            name=job.name,
            schedule=job.schedule,
            enabled=job.enabled,
            status=job.status,
            created_at=job.created_at.isoformat(),
            last_run=job.last_run.isoformat() if job.last_run else None,
            next_run=job.next_run.isoformat() if job.next_run else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=List[JobResponse])
async def list_training_jobs(
    scheduler: TrainingScheduler = Depends(get_scheduler)
):
    """Get list of all training jobs."""
    try:
        jobs = scheduler.get_jobs()
        
        return [
            JobResponse(
                job_id=job.job_id,
                name=job.name,
                schedule=job.schedule,
                enabled=job.enabled,
                status=job.status,
                created_at=job.created_at.isoformat(),
                last_run=job.last_run.isoformat() if job.last_run else None,
                next_run=job.next_run.isoformat() if job.next_run else None
            )
            for job in jobs
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_training_job(
    job_id: str,
    scheduler: TrainingScheduler = Depends(get_scheduler)
):
    """Get details of a specific training job."""
    try:
        job = scheduler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobResponse(
            job_id=job.job_id,
            name=job.name,
            schedule=job.schedule,
            enabled=job.enabled,
            status=job.status,
            created_at=job.created_at.isoformat(),
            last_run=job.last_run.isoformat() if job.last_run else None,
            next_run=job.next_run.isoformat() if job.next_run else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/trigger", response_model=TriggerJobResponse)
async def trigger_training_job(
    job_id: str,
    scheduler: TrainingScheduler = Depends(get_scheduler)
):
    """Manually trigger a training job."""
    try:
        execution_id = await scheduler.trigger_job(job_id)
        
        return TriggerJobResponse(
            execution_id=execution_id,
            message=f"Job {job_id} triggered successfully"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/pause")
async def pause_training_job(
    job_id: str,
    scheduler: TrainingScheduler = Depends(get_scheduler)
):
    """Pause a training job."""
    try:
        await scheduler.pause_job(job_id)
        return {"message": f"Job {job_id} paused successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/resume")
async def resume_training_job(
    job_id: str,
    scheduler: TrainingScheduler = Depends(get_scheduler)
):
    """Resume a paused training job."""
    try:
        await scheduler.resume_job(job_id)
        return {"message": f"Job {job_id} resumed successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def delete_training_job(
    job_id: str,
    scheduler: TrainingScheduler = Depends(get_scheduler)
):
    """Delete a training job."""
    try:
        await scheduler.remove_job(job_id)
        return {"message": f"Job {job_id} deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/executions/{execution_id}")
async def get_execution_status(execution_id: str):
    """Get status of a specific job execution."""
    try:
        # This would typically query execution status from database
        execution_status = {
            "execution_id": execution_id,
            "status": "running",
            "progress": 0.45,
            "started_at": "2025-06-29T10:00:00Z",
            "estimated_completion": "2025-06-29T14:00:00Z",
            "current_step": "train_model",
            "steps_completed": 3,
            "total_steps": 7,
            "logs": [
                {"timestamp": "2025-06-29T10:00:00Z", "level": "INFO", "message": "Execution started"},
                {"timestamp": "2025-06-29T10:15:00Z", "level": "INFO", "message": "Data preparation completed"},
                {"timestamp": "2025-06-29T10:30:00Z", "level": "INFO", "message": "Model initialization completed"}
            ]
        }
        
        return execution_status
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedule/validate")
async def validate_schedule(schedule: str):
    """Validate a schedule expression."""
    try:
        # Parse and validate the schedule
        from src.automation.scheduler import TrainingScheduler
        
        scheduler = TrainingScheduler()
        trigger = scheduler._parse_schedule(schedule)
        
        return {
            "valid": True,
            "schedule": schedule,
            "description": f"Schedule parsed successfully: {type(trigger).__name__}"
        }
    
    except Exception as e:
        return {
            "valid": False,
            "schedule": schedule,
            "error": str(e)
        }


@router.get("/templates")
async def get_job_templates():
    """Get predefined job templates for common training scenarios."""
    templates = {
        "basic_training": {
            "name": "Basic Model Training",
            "description": "Standard training pipeline for language models",
            "schedule": "cron:0 2 * * *",  # Daily at 2 AM
            "pipeline_config": {
                "data": {
                    "dataset_name": "training_data",
                    "preprocessing": {
                        "tokenization": True,
                        "normalization": True
                    }
                },
                "model": {
                    "type": "transformer",
                    "name": "base_model",
                    "config": {
                        "learning_rate": 0.001,
                        "batch_size": 32
                    }
                },
                "training": {
                    "epochs": 10,
                    "validation_split": 0.2,
                    "early_stopping": True
                },
                "evaluation": {
                    "metrics": ["accuracy", "f1_score"],
                    "performance_threshold": 0.8
                }
            }
        },
        "fine_tuning": {
            "name": "Model Fine-tuning",
            "description": "Fine-tune existing model with new data",
            "schedule": "interval:hours=6",
            "pipeline_config": {
                "data": {
                    "dataset_name": "fine_tune_data",
                    "preprocessing": {"minimal": True}
                },
                "model": {
                    "type": "transformer",
                    "name": "fine_tuned_model",
                    "base_model": "pretrained_model",
                    "config": {
                        "learning_rate": 0.0001,
                        "batch_size": 16
                    }
                },
                "training": {
                    "epochs": 3,
                    "freeze_layers": 8
                }
            }
        },
        "reinforcement_learning": {
            "name": "Reinforcement Learning",
            "description": "Reinforcement learning based on feedback",
            "schedule": "cron:0 */4 * * *",  # Every 4 hours
            "pipeline_config": {
                "training_type": "reinforcement",
                "data": {
                    "feedback_data": "user_feedback",
                    "reward_model": "reward_classifier"
                },
                "model": {
                    "type": "transformer",
                    "name": "rl_model",
                    "config": {
                        "learning_rate": 0.0001,
                        "ppo_config": {
                            "clip_ratio": 0.2,
                            "value_loss_coef": 0.5
                        }
                    }
                }
            }
        }
    }
    
    return {"templates": templates}


@router.get("/statistics")
async def get_automation_statistics(
    scheduler: TrainingScheduler = Depends(get_scheduler)
):
    """Get automation system statistics."""
    try:
        jobs = scheduler.get_jobs()
        
        stats = {
            "total_jobs": len(jobs),
            "active_jobs": len([j for j in jobs if j.enabled]),
            "running_jobs": len([j for j in jobs if j.status == "running"]),
            "scheduled_jobs": len([j for j in jobs if j.status == "scheduled"]),
            "completed_jobs": len([j for j in jobs if j.status == "completed"]),
            "failed_jobs": len([j for j in jobs if j.status == "failed"]),
            "jobs_by_schedule": {},
            "most_recent_execution": None
        }
        
        # Group jobs by schedule type
        for job in jobs:
            schedule_type = "cron" if job.schedule.startswith("cron:") else "interval"
            stats["jobs_by_schedule"][schedule_type] = stats["jobs_by_schedule"].get(schedule_type, 0) + 1
        
        # Find most recent execution
        recent_jobs = [j for j in jobs if j.last_run]
        if recent_jobs:
            most_recent = max(recent_jobs, key=lambda x: x.last_run)
            stats["most_recent_execution"] = {
                "job_id": most_recent.job_id,
                "job_name": most_recent.name,
                "executed_at": most_recent.last_run.isoformat()
            }
        
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
