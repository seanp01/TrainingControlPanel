"""
Tests for monitoring functionality.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.monitoring.metrics_collector import MetricsCollector


@pytest.fixture
def metrics_collector():
    """Create a metrics collector instance for testing."""
    return MetricsCollector()


@pytest.mark.asyncio
async def test_metrics_collector_start_stop(metrics_collector):
    """Test metrics collector start and stop."""
    await metrics_collector.start()
    assert metrics_collector.running
    
    await metrics_collector.stop()
    assert not metrics_collector.running


def test_record_training_job_metrics(metrics_collector):
    """Test recording training job metrics."""
    # Test starting a job
    metrics_collector.record_training_job_started("job_1", "Test Job")
    
    # Test completing a job
    metrics_collector.record_training_job_completed(
        "job_1", "Test Job", "completed", 3600.0
    )


def test_record_model_metrics(metrics_collector):
    """Test recording model performance metrics."""
    metrics_collector.record_model_metrics("test_model", "v1.0", 0.92)


def test_record_drift_score(metrics_collector):
    """Test recording drift scores."""
    metrics_collector.record_drift_score("test_model", 0.15)


def test_get_current_metrics(metrics_collector):
    """Test getting current system metrics."""
    with patch('psutil.cpu_percent', return_value=45.0), \
         patch('psutil.virtual_memory') as mock_memory, \
         patch('psutil.disk_usage') as mock_disk:
        
        mock_memory.return_value.percent = 60.0
        mock_memory.return_value.available = 8000000000
        
        mock_disk.return_value.used = 500000000000
        mock_disk.return_value.total = 1000000000000
        mock_disk.return_value.free = 500000000000
        
        metrics = metrics_collector.get_current_metrics()
        
        assert 'timestamp' in metrics
        assert 'system' in metrics
        assert 'training' in metrics
        assert metrics['system']['cpu_usage'] == 45.0


def test_prometheus_metrics(metrics_collector):
    """Test Prometheus metrics generation."""
    prometheus_metrics = metrics_collector.get_prometheus_metrics()
    assert isinstance(prometheus_metrics, str)
    assert len(prometheus_metrics) > 0
