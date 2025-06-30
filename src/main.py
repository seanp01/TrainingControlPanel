"""
LM Training Control Panel - Main Entry Point
"""
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import monitoring, automation, debugging, models
from src.core.config import settings
from src.core.logging_config import setup_logging
from src.automation.scheduler import TrainingScheduler
from src.monitoring.metrics_collector import MetricsCollector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize services
    logger.info("Starting LM Training Control Panel...")
    
    # Start metrics collector
    metrics_collector = MetricsCollector()
    await metrics_collector.start()
    app.state.metrics_collector = metrics_collector
    
    # Start training scheduler
    scheduler = TrainingScheduler()
    await scheduler.start()
    app.state.scheduler = scheduler
    
    logger.info("All services started successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down services...")
    await scheduler.stop()
    await metrics_collector.stop()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LM Training Control Panel",
        description="Automated Language Model Training Control Panel with Monitoring and Debugging",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["monitoring"])
    app.include_router(automation.router, prefix="/api/v1/automation", tags=["automation"])
    app.include_router(debugging.router, prefix="/api/v1/debugging", tags=["debugging"])
    app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
    
    return app


app = create_app()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LM Training Control Panel",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
