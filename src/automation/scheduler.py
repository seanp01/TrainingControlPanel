"""
Training scheduler for automated LM training workflows.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.core.config import settings
from src.automation.training_pipeline import TrainingPipeline
from src.monitoring.drift_detector import DriftDetector


logger = logging.getLogger(__name__)


class TrainingJob:
    """Represents a scheduled training job."""
    
    def __init__(
        self,
        job_id: str,
        name: str,
        pipeline_config: Dict[str, Any],
        schedule: str,
        enabled: bool = True
    ):
        self.job_id = job_id
        self.name = name
        self.pipeline_config = pipeline_config
        self.schedule = schedule
        self.enabled = enabled
        self.created_at = datetime.utcnow()
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self.status: str = "scheduled"


class TrainingScheduler:
    """Manages automated training schedules and execution."""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.jobs: Dict[str, TrainingJob] = {}
        self.pipeline = TrainingPipeline()
        self.drift_detector = DriftDetector()
        self.running_jobs: Dict[str, asyncio.Task] = {}
    
    async def start(self):
        """Start the scheduler."""
        logger.info("Starting training scheduler...")
        self.scheduler.start()
        
        # Schedule drift detection checks
        self.scheduler.add_job(
            self._check_drift,
            IntervalTrigger(minutes=30),
            id="drift_check",
            name="Drift Detection Check"
        )
        
        logger.info("Training scheduler started")
    
    async def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping training scheduler...")
        
        # Cancel running jobs
        for task in self.running_jobs.values():
            task.cancel()
        
        # Wait for jobs to complete
        if self.running_jobs:
            await asyncio.gather(*self.running_jobs.values(), return_exceptions=True)
        
        self.scheduler.shutdown()
        logger.info("Training scheduler stopped")
    
    async def add_job(
        self,
        name: str,
        pipeline_config: Dict[str, Any],
        schedule: str,
        job_id: Optional[str] = None
    ) -> str:
        """Add a new training job."""
        if job_id is None:
            job_id = str(uuid4())
        
        job = TrainingJob(job_id, name, pipeline_config, schedule)
        self.jobs[job_id] = job
        
        # Parse schedule and add to APScheduler
        trigger = self._parse_schedule(schedule)
        
        self.scheduler.add_job(
            self._execute_training_job,
            trigger,
            args=[job_id],
            id=job_id,
            name=name,
            max_instances=1
        )
        
        logger.info(f"Added training job: {name} ({job_id})")
        return job_id
    
    async def remove_job(self, job_id: str):
        """Remove a training job."""
        if job_id in self.jobs:
            self.scheduler.remove_job(job_id)
            del self.jobs[job_id]
            logger.info(f"Removed training job: {job_id}")
    
    async def pause_job(self, job_id: str):
        """Pause a training job."""
        if job_id in self.jobs:
            self.scheduler.pause_job(job_id)
            self.jobs[job_id].enabled = False
            logger.info(f"Paused training job: {job_id}")
    
    async def resume_job(self, job_id: str):
        """Resume a training job."""
        if job_id in self.jobs:
            self.scheduler.resume_job(job_id)
            self.jobs[job_id].enabled = True
            logger.info(f"Resumed training job: {job_id}")
    
    async def trigger_job(self, job_id: str) -> str:
        """Manually trigger a training job."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        execution_id = str(uuid4())
        task = asyncio.create_task(
            self._execute_training_job(job_id, execution_id)
        )
        self.running_jobs[execution_id] = task
        
        logger.info(f"Manually triggered job: {job_id} (execution: {execution_id})")
        return execution_id
    
    def get_jobs(self) -> List[TrainingJob]:
        """Get all training jobs."""
        return list(self.jobs.values())
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a specific training job."""
        return self.jobs.get(job_id)
    
    async def _execute_training_job(self, job_id: str, execution_id: Optional[str] = None):
        """Execute a training job."""
        if execution_id is None:
            execution_id = str(uuid4())
        
        job = self.jobs.get(job_id)
        if not job or not job.enabled:
            return
        
        job.status = "running"
        job.last_run = datetime.now()
        
        try:
            logger.info(f"Starting training job: {job.name} ({job_id})")
            
            # Execute the training pipeline
            result = await self.pipeline.execute(job.pipeline_config, execution_id)
            
            job.status = "completed"
            logger.info(f"Completed training job: {job.name} ({job_id})")
            
            # Check if reinforcement is needed based on results
            if result.get("needs_reinforcement", False):
                await self._schedule_reinforcement(job_id, result)
            
        except Exception as e:
            job.status = "failed"
            logger.error(f"Training job failed: {job.name} ({job_id}) - {str(e)}")
        
        finally:
            if execution_id in self.running_jobs:
                del self.running_jobs[execution_id]
    
    async def _check_drift(self):
        """Check for data/model drift and trigger retraining if needed."""
        logger.info("Checking for drift...")
        
        try:
            drift_results = await self.drift_detector.detect_drift()
            
            for model_name, drift_score in drift_results.items():
                if drift_score > settings.DRIFT_DETECTION_THRESHOLD:
                    logger.warning(f"Drift detected for {model_name}: {drift_score}")
                    await self._trigger_drift_response(model_name, drift_score)
        
        except Exception as e:
            logger.error(f"Drift detection failed: {str(e)}")
    
    async def _trigger_drift_response(self, model_name: str, drift_score: float):
        """Trigger response to detected drift."""
        # Find jobs related to this model
        related_jobs = [
            job for job in self.jobs.values()
            if job.pipeline_config.get("model_name") == model_name
        ]
        
        if related_jobs:
            # Trigger immediate retraining
            for job in related_jobs:
                await self.trigger_job(job.job_id)
                logger.info(f"Triggered drift response for {model_name}")
    
    async def _schedule_reinforcement(self, job_id: str, training_result: Dict[str, Any]):
        """Schedule reinforcement training based on training results."""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        # Create reinforcement job configuration
        reinforcement_config = job.pipeline_config.copy()
        reinforcement_config.update({
            "training_type": "reinforcement",
            "base_model": training_result.get("model_path"),
            "feedback_data": training_result.get("feedback_data")
        })
        
        # Schedule reinforcement job to run in 1 hour
        reinforcement_job_id = f"{job_id}_reinforcement_{int(datetime.utcnow().timestamp())}"
        
        self.scheduler.add_job(
            self._execute_training_job,
            "date",
            run_date=datetime.utcnow() + timedelta(hours=1),
            args=[reinforcement_job_id],
            id=reinforcement_job_id,
            name=f"Reinforcement for {job.name}"
        )
        
        logger.info(f"Scheduled reinforcement training for {job.name}")
    
    def _parse_schedule(self, schedule: str):
        """Parse schedule string into APScheduler trigger."""
        if schedule.startswith("cron:"):
            # Parse cron expression
            cron_expr = schedule[5:]
            parts = cron_expr.split()
            
            if len(parts) == 5:
                minute, hour, day, month, day_of_week = parts
                return CronTrigger(
                    minute=minute,
                    hour=hour,
                    day=day,
                    month=month,
                    day_of_week=day_of_week
                )
        
        elif schedule.startswith("interval:"):
            # Parse interval expression (e.g., "interval:hours=2")
            interval_expr = schedule[9:]
            kwargs = {}
            
            for part in interval_expr.split(","):
                key, value = part.split("=")
                kwargs[key.strip()] = int(value.strip())
            
            return IntervalTrigger(**kwargs)
        
        else:
            # Default to hourly
            return IntervalTrigger(hours=1)
