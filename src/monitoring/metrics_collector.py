"""
Metrics collector for monitoring training progress and system health.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

import psutil
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
from prometheus_client.exposition import generate_latest

from src.core.config import settings


logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and manages system and training metrics."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.running = False
        self.collection_task: Optional[asyncio.Task] = None
        
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        # Training metrics
        self.active_training_jobs = Gauge(
            'training_jobs_active',
            'Number of active training jobs',
            registry=self.registry
        )
        
        self.completed_training_jobs = Counter(
            'training_jobs_completed_total',
            'Total completed training jobs',
            ['status'],
            registry=self.registry
        )
        
        self.training_duration = Histogram(
            'training_job_duration_seconds',
            'Training job duration in seconds',
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Model accuracy score',
            ['model_name', 'version'],
            registry=self.registry
        )
        
        self.drift_score = Gauge(
            'model_drift_score',
            'Model drift detection score',
            ['model_name'],
            registry=self.registry
        )
        
        # API metrics
        self.api_requests = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.api_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Storage for historical data
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history_size = 10000
    
    async def start(self):
        """Start metrics collection."""
        if self.running:
            return
        
        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collector started")
    
    async def stop(self):
        """Stop metrics collection."""
        self.running = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metrics collector stopped")
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(settings.METRICS_COLLECTION_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.disk_usage.set(disk_percent)
            
            # Store in history
            timestamp = datetime.utcnow()
            metrics_snapshot = {
                'timestamp': timestamp.isoformat(),
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk_percent
            }
            
            self.metrics_history.append(metrics_snapshot)
            
            # Trim history if too large
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    def record_training_job_started(self, job_id: str, job_name: str):
        """Record that a training job has started."""
        self.active_training_jobs.inc()
        logger.debug(f"Training job started: {job_name} ({job_id})")
    
    def record_training_job_completed(self, job_id: str, job_name: str, status: str, duration: float):
        """Record that a training job has completed."""
        self.active_training_jobs.dec()
        self.completed_training_jobs.labels(status=status).inc()
        self.training_duration.observe(duration)
        logger.debug(f"Training job completed: {job_name} ({job_id}) - {status}")
    
    def record_model_metrics(self, model_name: str, version: str, accuracy: float):
        """Record model performance metrics."""
        self.model_accuracy.labels(model_name=model_name, version=version).set(accuracy)
        logger.debug(f"Model metrics recorded: {model_name} v{version} - accuracy: {accuracy}")
    
    def record_drift_score(self, model_name: str, score: float):
        """Record model drift score."""
        self.drift_score.labels(model_name=model_name).set(score)
        logger.debug(f"Drift score recorded: {model_name} - score: {score}")
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics."""
        self.api_requests.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
        self.api_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'system': {
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory.percent,
                    'memory_available': memory.available,
                    'disk_usage': (disk.used / disk.total) * 100,
                    'disk_free': disk.free
                },
                'training': {
                    'active_jobs': len(self._get_active_jobs()),
                    'total_completed': self._get_total_completed_jobs()
                }
            }
        except Exception as e:
            logger.error(f"Error getting current metrics: {str(e)}")
            return {}
    
    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history for the specified number of hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            metric for metric in self.metrics_history
            if datetime.fromisoformat(metric['timestamp']) > cutoff_time
        ]
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training activity summary."""
        # This would typically query the database for training job statistics
        return {
            'active_jobs': len(self._get_active_jobs()),
            'completed_today': self._get_jobs_completed_today(),
            'failed_today': self._get_jobs_failed_today(),
            'average_duration': self._get_average_job_duration(),
            'success_rate': self._get_success_rate()
        }
    
    def _get_active_jobs(self) -> List[str]:
        """Get list of currently active job IDs."""
        # Placeholder - would typically query job status from database
        return []
    
    def _get_total_completed_jobs(self) -> int:
        """Get total number of completed jobs."""
        # Placeholder - would typically query database
        return 0
    
    def _get_jobs_completed_today(self) -> int:
        """Get number of jobs completed today."""
        # Placeholder - would typically query database
        return 0
    
    def _get_jobs_failed_today(self) -> int:
        """Get number of jobs failed today."""
        # Placeholder - would typically query database
        return 0
    
    def _get_average_job_duration(self) -> float:
        """Get average job duration in seconds."""
        # Placeholder - would typically calculate from database
        return 0.0
    
    def _get_success_rate(self) -> float:
        """Get training job success rate as percentage."""
        # Placeholder - would typically calculate from database
        return 0.0
