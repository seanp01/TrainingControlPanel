"""
Tests for the training scheduler functionality.
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from src.automation.scheduler import TrainingScheduler, TrainingJob


@pytest.fixture
async def scheduler():
    """Create a scheduler instance for testing."""
    scheduler = TrainingScheduler()
    yield scheduler
    await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_start_stop(scheduler):
    """Test scheduler start and stop functionality."""
    await scheduler.start()
    assert scheduler.scheduler.running
    
    await scheduler.stop()
    assert not scheduler.scheduler.running


@pytest.mark.asyncio
async def test_add_training_job(scheduler):
    """Test adding a new training job."""
    await scheduler.start()
    
    job_config = {
        "model": {"type": "transformer", "name": "test_model"},
        "training": {"epochs": 5, "batch_size": 16}
    }
    
    job_id = await scheduler.add_job(
        name="Test Job",
        pipeline_config=job_config,
        schedule="interval:hours=1"
    )
    
    assert job_id is not None
    job = scheduler.get_job(job_id)
    assert job is not None
    assert job.name == "Test Job"
    assert job.pipeline_config == job_config


@pytest.mark.asyncio
async def test_pause_resume_job(scheduler):
    """Test pausing and resuming jobs."""
    await scheduler.start()
    
    job_id = await scheduler.add_job(
        name="Test Job",
        pipeline_config={},
        schedule="interval:hours=1"
    )
    
    # Pause job
    await scheduler.pause_job(job_id)
    job = scheduler.get_job(job_id)
    assert not job.enabled
    
    # Resume job
    await scheduler.resume_job(job_id)
    job = scheduler.get_job(job_id)
    assert job.enabled


@pytest.mark.asyncio
async def test_remove_job(scheduler):
    """Test removing a job."""
    await scheduler.start()
    
    job_id = await scheduler.add_job(
        name="Test Job",
        pipeline_config={},
        schedule="interval:hours=1"
    )
    
    await scheduler.remove_job(job_id)
    job = scheduler.get_job(job_id)
    assert job is None


def test_training_job_creation():
    """Test TrainingJob creation."""
    job = TrainingJob(
        job_id="test_123",
        name="Test Job",
        pipeline_config={"test": "config"},
        schedule="cron:0 * * * *"
    )
    
    assert job.job_id == "test_123"
    assert job.name == "Test Job"
    assert job.pipeline_config == {"test": "config"}
    assert job.schedule == "cron:0 * * * *"
    assert job.enabled is True
    assert job.status == "scheduled"
