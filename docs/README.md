# LM Training Control Panel Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Getting Started](#getting-started)
3. [API Reference](#api-reference)
4. [Automation System](#automation-system)
5. [Monitoring & Debugging](#monitoring--debugging)
6. [Configuration](#configuration)
7. [Development Guide](#development-guide)

## Architecture Overview

The LM Training Control Panel is built with a modular architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web UI        │    │   FastAPI       │    │   LangGraph     │
│   (Frontend)    │◄──►│   (API Layer)   │◄──►│   (Workflows)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Automation    │    │   Data & Models │
│   (Metrics)     │    │   (Scheduling)  │    │   (DVC/MLflow)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

- **Automation Engine**: Manages training schedules, feedback loops, and drift response
- **Monitoring System**: Tracks training progress, system health, and model performance
- **Debugging Tools**: Provides debugging capabilities for training pipelines
- **Data Management**: Handles data versioning with DVC integration
- **Model Management**: Manages model lifecycle with MLflow tracking

## Getting Started

### Prerequisites

- Python 3.9+
- PostgreSQL
- Redis
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd TrainingControlPanel
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up configuration:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Initialize DVC:
   ```bash
   dvc init
   ```

6. Start the application:
   ```bash
   python -m src.main
   ```

The application will be available at `http://localhost:8000`

### API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

## API Reference

### Monitoring Endpoints

- `GET /api/v1/monitoring/health` - Health check
- `GET /api/v1/monitoring/metrics` - Current system metrics
- `GET /api/v1/monitoring/training/jobs` - Training job status
- `GET /api/v1/monitoring/drift/summary` - Drift detection summary

### Automation Endpoints

- `POST /api/v1/automation/jobs` - Create training job
- `GET /api/v1/automation/jobs` - List training jobs
- `POST /api/v1/automation/jobs/{id}/trigger` - Trigger job manually
- `POST /api/v1/automation/jobs/{id}/pause` - Pause job

### Debugging Endpoints

- `POST /api/v1/debugging/sessions` - Create debug session
- `GET /api/v1/debugging/logs` - Get training logs
- `GET /api/v1/debugging/profiling/{execution_id}` - Get profiling data

### Models Endpoints

- `GET /api/v1/models/` - List all models
- `GET /api/v1/models/{id}` - Get model details
- `POST /api/v1/models/{id}/promote` - Promote model to environment

## Automation System

### Scheduling

The automation system supports two types of schedules:

1. **Cron expressions**: `cron:0 2 * * *` (daily at 2 AM)
2. **Interval schedules**: `interval:hours=6` (every 6 hours)

### Training Pipelines

Training pipelines are built using LangGraph workflows with these steps:

1. **prepare_data** - Data loading and preprocessing
2. **validate_data** - Data quality validation
3. **initialize_model** - Model setup and configuration
4. **train_model** - Model training execution
5. **evaluate_model** - Performance evaluation
6. **save_model** - Model persistence
7. **update_tracking** - Experiment tracking updates

### Feedback Loops

The system implements automated feedback loops:

- **Drift Detection**: Monitors model performance degradation
- **Reinforcement Learning**: Incorporates user feedback for model improvement
- **Auto-retraining**: Triggers retraining when drift is detected

### Example Job Configuration

```json
{
  "name": "Daily Model Training",
  "schedule": "cron:0 2 * * *",
  "pipeline_config": {
    "data": {
      "dataset_name": "training_data_v3",
      "preprocessing": {
        "tokenization": true,
        "normalization": true
      }
    },
    "model": {
      "type": "transformer",
      "name": "main_model",
      "config": {
        "learning_rate": 0.001,
        "batch_size": 32
      }
    },
    "training": {
      "epochs": 10,
      "validation_split": 0.2
    },
    "evaluation": {
      "performance_threshold": 0.8
    }
  }
}
```

## Monitoring & Debugging

### Metrics Collection

The system collects:

- **System Metrics**: CPU, memory, disk usage
- **Training Metrics**: Job progress, model performance
- **API Metrics**: Request counts, response times
- **Drift Scores**: Model performance degradation

### Drift Detection

Uses Evidently AI for:

- **Data Drift**: Changes in input data distribution
- **Target Drift**: Changes in target variable distribution
- **Model Performance**: Accuracy degradation over time

### Debugging Features

- **Interactive Debugging**: Step-by-step execution debugging
- **Performance Profiling**: Training performance analysis
- **Log Analysis**: Advanced log searching and filtering
- **Error Patterns**: Common error detection and suggestions

## Configuration

### Environment Variables

Key environment variables (see `.env.example`):

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `OPENAI_API_KEY`: OpenAI API key (optional)
- `DRIFT_DETECTION_THRESHOLD`: Drift alert threshold

### DVC Configuration

Configure data versioning in `dvc.yaml`:

```yaml
stages:
  prepare_data:
    cmd: python -m src.data.prepare_data
    deps:
    - src/data/prepare_data.py
    outs:
    - data/processed/training_data.parquet
```

### MLflow Configuration

MLflow tracks:

- **Experiments**: Training runs and parameters
- **Models**: Model artifacts and metadata
- **Metrics**: Training and evaluation metrics

## Development Guide

### Project Structure

```
src/
├── api/              # FastAPI endpoints
├── automation/       # Training scheduling and automation
├── monitoring/       # Metrics and drift detection
├── debugging/        # Debugging tools
├── models/          # Model management
├── data/            # Data management
└── core/            # Core configuration

config/              # Configuration files
data/                # Training data (DVC tracked)
models/              # Model artifacts (DVC tracked)
experiments/         # MLflow experiments
tests/               # Test suites
docs/                # Documentation
```

### Adding New Features

1. **Create module** in appropriate `src/` subdirectory
2. **Add API endpoints** in `src/api/`
3. **Write tests** in `tests/`
4. **Update documentation** in `docs/`

### Testing

Run tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

Format code:
```bash
black src/ tests/
isort src/ tests/
```

Lint code:
```bash
flake8 src/ tests/
mypy src/
```

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pre-commit install
```

This ensures code quality checks run before commits.

### Deployment

The application can be deployed using:

- **Docker**: Containerized deployment
- **Kubernetes**: Scalable cloud deployment
- **Docker Compose**: Local development environment

See deployment documentation for detailed instructions.
