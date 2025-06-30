"""
Drift detection system for monitoring model performance degradation.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift

from src.core.config import settings
from src.data.data_manager import DataManager


logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data and model drift for automated retraining triggers."""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.drift_history: Dict[str, List[Dict[str, Any]]] = {}
        self.baseline_data: Dict[str, Any] = {}
    
    async def detect_drift(self) -> Dict[str, float]:
        """Detect drift across all monitored models."""
        drift_results = {}
        
        try:
            # Get list of models to monitor
            models = await self._get_monitored_models()
            
            for model_name in models:
                drift_score = await self._detect_model_drift(model_name)
                drift_results[model_name] = drift_score
                
                # Store drift history
                self._store_drift_result(model_name, drift_score)
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Drift detection failed: {str(e)}")
            return {}
    
    async def _detect_model_drift(self, model_name: str) -> float:
        """Detect drift for a specific model."""
        try:
            # Get reference (baseline) data
            reference_data = await self._get_reference_data(model_name)
            if reference_data is None:
                logger.warning(f"No reference data for model {model_name}")
                return 0.0
            
            # Get current production data
            current_data = await self._get_current_data(model_name)
            if current_data is None:
                logger.warning(f"No current data for model {model_name}")
                return 0.0
            
            # Perform drift detection
            drift_score = await self._calculate_drift_score(
                reference_data, current_data, model_name
            )
            
            logger.info(f"Drift score for {model_name}: {drift_score}")
            return drift_score
            
        except Exception as e:
            logger.error(f"Drift detection failed for {model_name}: {str(e)}")
            return 0.0
    
    async def _calculate_drift_score(
        self, 
        reference_data: Any, 
        current_data: Any, 
        model_name: str
    ) -> float:
        """Calculate drift score using Evidently."""
        try:
            # Create column mapping
            column_mapping = ColumnMapping()
            
            # Detect feature drift
            data_drift_report = Report(metrics=[DataDriftPreset()])
            data_drift_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            # Extract drift metrics
            report_dict = data_drift_report.as_dict()
            metrics = report_dict.get('metrics', [])
            
            # Calculate overall drift score
            drift_scores = []
            
            for metric in metrics:
                if metric.get('metric') == 'DatasetDriftMetric':
                    result = metric.get('result', {})
                    drift_share = result.get('drift_share', 0.0)
                    drift_scores.append(drift_share)
            
            # Return maximum drift score
            return max(drift_scores) if drift_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating drift score for {model_name}: {str(e)}")
            return 0.0
    
    async def _get_monitored_models(self) -> List[str]:
        """Get list of models to monitor for drift."""
        # This would typically query a database or config file
        # For now, return a placeholder list
        return ["main_model", "assistant_model", "classifier_model"]
    
    async def _get_reference_data(self, model_name: str) -> Optional[Any]:
        """Get reference (baseline) data for drift comparison."""
        try:
            # Check if we have cached baseline data
            if model_name in self.baseline_data:
                return self.baseline_data[model_name]
            
            # Load baseline data from storage
            baseline_path = Path(settings.DATA_DIR) / "baselines" / f"{model_name}_baseline.parquet"
            
            if baseline_path.exists():
                baseline_data = await self.data_manager.load_data(str(baseline_path))
                self.baseline_data[model_name] = baseline_data
                return baseline_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading reference data for {model_name}: {str(e)}")
            return None
    
    async def _get_current_data(self, model_name: str) -> Optional[Any]:
        """Get current production data for drift comparison."""
        try:
            # Get recent production data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)  # Last 24 hours
            
            current_data = await self.data_manager.get_production_data(
                model_name=model_name,
                start_time=start_time,
                end_time=end_time
            )
            
            return current_data
            
        except Exception as e:
            logger.error(f"Error loading current data for {model_name}: {str(e)}")
            return None
    
    def _store_drift_result(self, model_name: str, drift_score: float):
        """Store drift detection result in history."""
        if model_name not in self.drift_history:
            self.drift_history[model_name] = []
        
        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'drift_score': drift_score,
            'threshold': settings.DRIFT_DETECTION_THRESHOLD,
            'alert_triggered': drift_score > settings.DRIFT_DETECTION_THRESHOLD
        }
        
        self.drift_history[model_name].append(result)
        
        # Keep only last 1000 results
        if len(self.drift_history[model_name]) > 1000:
            self.drift_history[model_name] = self.drift_history[model_name][-1000:]
    
    async def set_baseline(self, model_name: str, data: Any):
        """Set baseline data for a model."""
        try:
            # Store baseline data
            self.baseline_data[model_name] = data
            
            # Save to disk
            baseline_dir = Path(settings.DATA_DIR) / "baselines"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            
            baseline_path = baseline_dir / f"{model_name}_baseline.parquet"
            await self.data_manager.save_data(data, str(baseline_path))
            
            logger.info(f"Baseline data set for model {model_name}")
            
        except Exception as e:
            logger.error(f"Error setting baseline for {model_name}: {str(e)}")
            raise
    
    def get_drift_history(self, model_name: str, hours: int = 168) -> List[Dict[str, Any]]:
        """Get drift history for a model (default: last week)."""
        if model_name not in self.drift_history:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            result for result in self.drift_history[model_name]
            if datetime.fromisoformat(result['timestamp']) > cutoff_time
        ]
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift status across all models."""
        summary = {
            'models_monitored': len(self.drift_history),
            'models_with_drift': 0,
            'highest_drift_score': 0.0,
            'models_status': {}
        }
        
        for model_name, history in self.drift_history.items():
            if history:
                latest_result = history[-1]
                drift_score = latest_result['drift_score']
                
                summary['models_status'][model_name] = {
                    'latest_drift_score': drift_score,
                    'alert_triggered': latest_result['alert_triggered'],
                    'last_checked': latest_result['timestamp']
                }
                
                if drift_score > settings.DRIFT_DETECTION_THRESHOLD:
                    summary['models_with_drift'] += 1
                
                if drift_score > summary['highest_drift_score']:
                    summary['highest_drift_score'] = drift_score
        
        return summary
    
    async def run_drift_tests(self, model_name: str) -> Dict[str, Any]:
        """Run comprehensive drift tests for a model."""
        try:
            reference_data = await self._get_reference_data(model_name)
            current_data = await self._get_current_data(model_name)
            
            if reference_data is None or current_data is None:
                return {'error': 'Missing reference or current data'}
            
            # Create test suite
            tests = TestSuite(tests=[
                TestColumnDrift(),
            ])
            
            # Run tests
            tests.run(
                reference_data=reference_data,
                current_data=current_data
            )
            
            # Get test results
            test_results = tests.as_dict()
            
            return {
                'model_name': model_name,
                'test_results': test_results,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error running drift tests for {model_name}: {str(e)}")
            return {'error': str(e)}
