"""
Core configuration settings for the LM Training Control Panel.
"""
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "LM Training Control Panel"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/lm_training_db"
    
    # Redis (for Celery)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    
    # DVC
    DVC_REMOTE: str = "s3://your-bucket/dvc-storage"
    
    # OpenAI API
    OPENAI_API_KEY: Optional[str] = None
    
    # Monitoring
    METRICS_COLLECTION_INTERVAL: int = 30  # seconds
    DRIFT_DETECTION_THRESHOLD: float = 0.1
    
    # Training
    MAX_CONCURRENT_TRAINING_JOBS: int = 3
    TRAINING_CHECKPOINT_INTERVAL: int = 100  # steps
    
    # Paths
    DATA_DIR: str = "data"
    MODEL_DIR: str = "models"
    EXPERIMENT_DIR: str = "experiments"
    LOG_DIR: str = "logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
