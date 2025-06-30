"""
Training pipeline for automated LM training with LangGraph integration.
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from src.core.config import settings
from src.models.training_state import TrainingState
from src.data.data_manager import DataManager
from src.models.model_manager import ModelManager


logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Manages the automated training pipeline using LangGraph."""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        
        # Create the training workflow graph
        self.workflow = self._create_training_workflow()
    
    def _create_training_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for training."""
        
        workflow = StateGraph(TrainingState)
        
        # Add nodes for each training step
        workflow.add_node("prepare_data", self._prepare_data)
        workflow.add_node("validate_data", self._validate_data)
        workflow.add_node("initialize_model", self._initialize_model)
        workflow.add_node("train_model", self._train_model)
        workflow.add_node("evaluate_model", self._evaluate_model)
        workflow.add_node("save_model", self._save_model)
        workflow.add_node("update_tracking", self._update_tracking)
        
        # Define the workflow edges
        workflow.set_entry_point("prepare_data")
        
        workflow.add_edge("prepare_data", "validate_data")
        workflow.add_edge("validate_data", "initialize_model")
        workflow.add_edge("initialize_model", "train_model")
        workflow.add_edge("train_model", "evaluate_model")
        workflow.add_edge("evaluate_model", "save_model")
        workflow.add_edge("save_model", "update_tracking")
        workflow.add_edge("update_tracking", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "validate_data",
            self._should_continue_after_validation,
            {"continue": "initialize_model", "stop": END}
        )
        
        workflow.add_conditional_edges(
            "evaluate_model",
            self._should_continue_after_evaluation,
            {"continue": "save_model", "retrain": "train_model", "stop": END}
        )
        
        return workflow.compile()
    
    async def execute(self, config: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Execute the training pipeline."""
        logger.info(f"Starting training pipeline execution: {execution_id}")
        
        # Initialize state
        initial_state = TrainingState(
            execution_id=execution_id,
            config=config,
            started_at=datetime.utcnow(),
            status="running"
        )
        
        try:
            # Execute the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            logger.info(f"Training pipeline completed: {execution_id}")
            
            return {
                "execution_id": execution_id,
                "status": final_state.status,
                "model_path": final_state.model_path,
                "metrics": final_state.metrics,
                "needs_reinforcement": final_state.needs_reinforcement,
                "feedback_data": final_state.feedback_data,
                "completed_at": final_state.completed_at
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {execution_id} - {str(e)}")
            raise
    
    async def _prepare_data(self, state: TrainingState) -> TrainingState:
        """Prepare training data."""
        logger.info(f"Preparing data for {state.execution_id}")
        
        try:
            # Load and prepare data based on configuration
            data_config = state.config.get("data", {})
            
            data_path = await self.data_manager.prepare_training_data(
                dataset_name=data_config.get("dataset_name"),
                version=data_config.get("version"),
                preprocessing=data_config.get("preprocessing", {})
            )
            
            state.data_path = data_path
            state.status = "data_prepared"
            state.messages.append(HumanMessage(content=f"Data prepared at {data_path}"))
            
            return state
            
        except Exception as e:
            state.status = "failed"
            state.error = str(e)
            state.messages.append(AIMessage(content=f"Data preparation failed: {str(e)}"))
            raise
    
    async def _validate_data(self, state: TrainingState) -> TrainingState:
        """Validate the prepared data."""
        logger.info(f"Validating data for {state.execution_id}")
        
        try:
            validation_results = await self.data_manager.validate_data(state.data_path)
            
            state.data_validation = validation_results
            state.status = "data_validated"
            state.messages.append(
                AIMessage(content=f"Data validation completed: {validation_results}")
            )
            
            return state
            
        except Exception as e:
            state.status = "failed"
            state.error = str(e)
            state.messages.append(AIMessage(content=f"Data validation failed: {str(e)}"))
            raise
    
    async def _initialize_model(self, state: TrainingState) -> TrainingState:
        """Initialize the model for training."""
        logger.info(f"Initializing model for {state.execution_id}")
        
        try:
            model_config = state.config.get("model", {})
            
            model = await self.model_manager.initialize_model(
                model_type=model_config.get("type"),
                model_name=model_config.get("name"),
                config=model_config
            )
            
            state.model = model
            state.status = "model_initialized"
            state.messages.append(
                AIMessage(content=f"Model initialized: {model_config.get('name')}")
            )
            
            return state
            
        except Exception as e:
            state.status = "failed"
            state.error = str(e)
            state.messages.append(AIMessage(content=f"Model initialization failed: {str(e)}"))
            raise
    
    async def _train_model(self, state: TrainingState) -> TrainingState:
        """Train the model."""
        logger.info(f"Training model for {state.execution_id}")
        
        try:
            training_config = state.config.get("training", {})
            
            training_results = await self.model_manager.train_model(
                model=state.model,
                data_path=state.data_path,
                config=training_config,
                execution_id=state.execution_id
            )
            
            state.training_results = training_results
            state.status = "training_completed"
            state.messages.append(
                AIMessage(content=f"Training completed: {training_results}")
            )
            
            return state
            
        except Exception as e:
            state.status = "failed"
            state.error = str(e)
            state.messages.append(AIMessage(content=f"Training failed: {str(e)}"))
            raise
    
    async def _evaluate_model(self, state: TrainingState) -> TrainingState:
        """Evaluate the trained model."""
        logger.info(f"Evaluating model for {state.execution_id}")
        
        try:
            evaluation_config = state.config.get("evaluation", {})
            
            metrics = await self.model_manager.evaluate_model(
                model=state.model,
                data_path=state.data_path,
                config=evaluation_config
            )
            
            state.metrics = metrics
            state.status = "evaluation_completed"
            state.messages.append(
                AIMessage(content=f"Evaluation completed: {metrics}")
            )
            
            # Determine if reinforcement is needed
            performance_threshold = evaluation_config.get("performance_threshold", 0.8)
            current_performance = metrics.get("accuracy", 0.0)
            
            if current_performance < performance_threshold:
                state.needs_reinforcement = True
                state.feedback_data = await self._collect_feedback_data(state)
            
            return state
            
        except Exception as e:
            state.status = "failed"
            state.error = str(e)
            state.messages.append(AIMessage(content=f"Evaluation failed: {str(e)}"))
            raise
    
    async def _save_model(self, state: TrainingState) -> TrainingState:
        """Save the trained model."""
        logger.info(f"Saving model for {state.execution_id}")
        
        try:
            model_path = await self.model_manager.save_model(
                model=state.model,
                execution_id=state.execution_id,
                metrics=state.metrics
            )
            
            state.model_path = model_path
            state.status = "model_saved"
            state.messages.append(
                AIMessage(content=f"Model saved at {model_path}")
            )
            
            return state
            
        except Exception as e:
            state.status = "failed"
            state.error = str(e)
            state.messages.append(AIMessage(content=f"Model saving failed: {str(e)}"))
            raise
    
    async def _update_tracking(self, state: TrainingState) -> TrainingState:
        """Update experiment tracking."""
        logger.info(f"Updating tracking for {state.execution_id}")
        
        try:
            # Update MLflow tracking
            await self.model_manager.update_experiment_tracking(
                execution_id=state.execution_id,
                config=state.config,
                metrics=state.metrics,
                model_path=state.model_path
            )
            
            state.status = "completed"
            state.completed_at = datetime.utcnow()
            state.messages.append(
                AIMessage(content="Experiment tracking updated successfully")
            )
            
            return state
            
        except Exception as e:
            state.status = "failed"
            state.error = str(e)
            state.messages.append(AIMessage(content=f"Tracking update failed: {str(e)}"))
            raise
    
    def _should_continue_after_validation(self, state: TrainingState) -> str:
        """Determine if training should continue after data validation."""
        if state.status == "failed":
            return "stop"
        
        validation_results = state.data_validation or {}
        if not validation_results.get("is_valid", False):
            return "stop"
        
        return "continue"
    
    def _should_continue_after_evaluation(self, state: TrainingState) -> str:
        """Determine next step after model evaluation."""
        if state.status == "failed":
            return "stop"
        
        metrics = state.metrics or {}
        min_accuracy = state.config.get("evaluation", {}).get("min_accuracy", 0.5)
        
        accuracy = metrics.get("accuracy", 0.0)
        
        if accuracy < min_accuracy:
            # Check if we've already retrained
            retrain_count = state.config.get("_retrain_count", 0)
            max_retrains = state.config.get("training", {}).get("max_retrains", 2)
            
            if retrain_count < max_retrains:
                state.config["_retrain_count"] = retrain_count + 1
                return "retrain"
            else:
                return "stop"
        
        return "continue"
    
    async def _collect_feedback_data(self, state: TrainingState) -> Dict[str, Any]:
        """Collect feedback data for reinforcement learning."""
        # This would typically involve collecting user feedback,
        # analyzing model outputs, or gathering additional training data
        
        feedback_data = {
            "low_confidence_samples": [],
            "error_patterns": [],
            "suggested_improvements": []
        }
        
        # Placeholder for actual feedback collection logic
        logger.info(f"Collected feedback data for {state.execution_id}")
        
        return feedback_data
