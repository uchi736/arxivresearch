"""
State Management System for arXiv Research Agent

This module provides an improved state management system with:
- Immutability guarantees
- Type safety through Pydantic
- State validation
- Transaction-like updates
"""

from typing import Dict, List, Optional, Any, Union
from copy import deepcopy
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from src.core.models import (
    ResearchPlan, ImprovedResearchPlan, SearchQuery,
    PaperMetadata, OchiaiFormatAdvanced
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StateValidationError(Exception):
    """Raised when state validation fails"""
    pass


class WorkflowState(BaseModel):
    """
    Immutable workflow state with Pydantic validation
    """
    # Core fields
    initial_query: str
    analysis_mode: str = "advanced_moderate"
    
    # Research planning
    research_plan: Optional[Dict[str, Any]] = None
    improved_research_plan: Optional[Dict[str, Any]] = None
    token_budget: int = Field(default=30000, ge=1000, le=100000)
    
    # Search and papers
    search_queries: List[Dict[str, Any]] = Field(default_factory=list)
    found_papers: List[Dict[str, Any]] = Field(default_factory=list)
    analyzed_papers: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Results
    final_report: Optional[str] = None
    total_tokens_used: int = Field(default=0, ge=0)
    
    # Progress tracking
    progress_tracker: Optional[Dict[str, Any]] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        # Prevent mutation after creation
        frozen = True
        # Allow arbitrary types for compatibility
        arbitrary_types_allowed = True
    
    @field_validator('analysis_mode')
    @classmethod
    def validate_analysis_mode(cls, v):
        valid_modes = ["advanced_shallow", "advanced_moderate", "advanced_deep"]
        if v not in valid_modes:
            raise ValueError(f"Invalid analysis mode: {v}. Must be one of {valid_modes}")
        return v
    
    @field_validator('search_queries', 'found_papers', 'analyzed_papers')
    @classmethod
    def validate_list_items(cls, v):
        """Ensure list items are dictionaries"""
        for item in v:
            if not isinstance(item, dict):
                raise ValueError(f"List items must be dictionaries, got {type(item)}")
        return v
    
    def update(self, **kwargs) -> 'WorkflowState':
        """
        Create a new state with updated fields (immutable update)
        
        Args:
            **kwargs: Fields to update
            
        Returns:
            New WorkflowState instance with updates applied
        """
        # Get current data as dict
        current_data = self.model_dump()
        
        # Apply updates
        for key, value in kwargs.items():
            if key not in current_data:
                raise ValueError(f"Unknown field: {key}")
            current_data[key] = value
        
        # Update timestamp
        current_data['updated_at'] = datetime.now()
        
        # Create new instance
        return WorkflowState(**current_data)
    
    def validate_transition(self, next_step: str) -> bool:
        """
        Validate if transition to next step is allowed
        
        Args:
            next_step: Name of the next workflow step
            
        Returns:
            True if transition is valid
            
        Raises:
            StateValidationError if transition is invalid
        """
        # Define valid transitions
        transitions = {
            "plan_research": lambda s: s.initial_query is not None,
            "generate_queries": lambda s: s.research_plan is not None,
            "search_papers": lambda s: len(s.search_queries) > 0,
            "process_fulltext": lambda s: len(s.found_papers) > 0,
            "analyze": lambda s: True,  # Can always analyze
            "generate_report": lambda s: len(s.analyzed_papers) > 0,
            "save_results": lambda s: s.final_report is not None
        }
        
        if next_step not in transitions:
            raise StateValidationError(f"Unknown step: {next_step}")
        
        validator = transitions[next_step]
        if not validator(self):
            raise StateValidationError(
                f"Cannot transition to {next_step} - prerequisites not met"
            )
        
        return True
    
    def to_langgraph_state(self) -> Dict[str, Any]:
        """
        Convert to LangGraph-compatible state dictionary
        
        Returns:
            Dictionary compatible with LangGraph StateGraph
        """
        # Exclude Pydantic-specific fields
        data = self.model_dump()
        data.pop('created_at', None)
        data.pop('updated_at', None)
        return data


class StateManager:
    """
    Manages workflow state with transaction-like semantics
    """
    
    def __init__(self, initial_state: Optional[WorkflowState] = None):
        """
        Initialize state manager
        
        Args:
            initial_state: Initial workflow state
        """
        self._state = initial_state
        self._history: List[WorkflowState] = []
        if initial_state:
            self._history.append(initial_state)
    
    @property
    def current_state(self) -> Optional[WorkflowState]:
        """Get current state (read-only)"""
        return self._state
    
    def initialize(self, initial_query: str, analysis_mode: str = "advanced_moderate") -> WorkflowState:
        """
        Initialize new workflow state
        
        Args:
            initial_query: User's research query
            analysis_mode: Analysis depth mode
            
        Returns:
            Initialized WorkflowState
        """
        self._state = WorkflowState(
            initial_query=initial_query,
            analysis_mode=analysis_mode
        )
        self._history = [self._state]
        logger.info(f"Initialized workflow state for query: {initial_query}")
        return self._state
    
    def update(self, **kwargs) -> WorkflowState:
        """
        Update state with validation
        
        Args:
            **kwargs: Fields to update
            
        Returns:
            Updated WorkflowState
            
        Raises:
            StateValidationError if update is invalid
        """
        if not self._state:
            raise StateValidationError("State not initialized")
        
        # Create new state
        new_state = self._state.update(**kwargs)
        
        # Add to history
        self._history.append(new_state)
        self._state = new_state
        
        logger.debug(f"State updated: {list(kwargs.keys())}")
        return new_state
    
    def batch_update(self, updates: Dict[str, Any]) -> WorkflowState:
        """
        Perform multiple updates atomically
        
        Args:
            updates: Dictionary of updates to apply
            
        Returns:
            Updated WorkflowState
            
        Raises:
            StateValidationError if any update fails
        """
        if not self._state:
            raise StateValidationError("State not initialized")
        
        try:
            # Try to create new state with all updates
            new_state = self._state.update(**updates)
            
            # If successful, commit
            self._history.append(new_state)
            self._state = new_state
            
            logger.debug(f"Batch update successful: {len(updates)} fields")
            return new_state
            
        except Exception as e:
            logger.error(f"Batch update failed: {e}")
            raise StateValidationError(f"Batch update failed: {e}")
    
    def validate_transition(self, next_step: str) -> bool:
        """
        Validate workflow transition
        
        Args:
            next_step: Next workflow step
            
        Returns:
            True if transition is valid
        """
        if not self._state:
            raise StateValidationError("State not initialized")
        
        return self._state.validate_transition(next_step)
    
    def get_history(self) -> List[WorkflowState]:
        """Get state history (read-only)"""
        return self._history.copy()
    
    def rollback(self, steps: int = 1) -> Optional[WorkflowState]:
        """
        Rollback to previous state
        
        Args:
            steps: Number of steps to rollback
            
        Returns:
            Previous state or None if cannot rollback
        """
        if len(self._history) <= steps:
            logger.warning(f"Cannot rollback {steps} steps - insufficient history")
            return None
        
        # Remove recent states
        for _ in range(steps):
            self._history.pop()
        
        # Set state to last in history
        self._state = self._history[-1] if self._history else None
        logger.info(f"Rolled back {steps} steps")
        
        return self._state
    
    def export_for_langgraph(self) -> Dict[str, Any]:
        """
        Export current state for LangGraph
        
        Returns:
            LangGraph-compatible state dictionary
        """
        if not self._state:
            return {}
        
        return self._state.to_langgraph_state()
    
    def import_from_langgraph(self, langgraph_state: Dict[str, Any]) -> WorkflowState:
        """
        Import state from LangGraph
        
        Args:
            langgraph_state: State dictionary from LangGraph
            
        Returns:
            Imported WorkflowState
        """
        # Add timestamps if missing
        if 'created_at' not in langgraph_state:
            langgraph_state['created_at'] = datetime.now()
        if 'updated_at' not in langgraph_state:
            langgraph_state['updated_at'] = datetime.now()
        
        # Create new state
        self._state = WorkflowState(**langgraph_state)
        self._history = [self._state]
        
        logger.info("Imported state from LangGraph")
        return self._state


# Utility functions for node integration
def create_state_updater(state_manager: StateManager):
    """
    Create a state updater function for nodes
    
    Args:
        state_manager: StateManager instance
        
    Returns:
        Function that updates state and returns LangGraph-compatible dict
    """
    def update_state(**kwargs) -> Dict[str, Any]:
        """Update state and return LangGraph-compatible dictionary"""
        state_manager.update(**kwargs)
        return state_manager.export_for_langgraph()
    
    return update_state


def validate_node_transition(state_manager: StateManager, node_name: str):
    """
    Decorator to validate node transitions
    
    Args:
        state_manager: StateManager instance
        node_name: Name of the node
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Validate transition
            if not state_manager.validate_transition(node_name):
                raise StateValidationError(f"Invalid transition to {node_name}")
            
            # Execute node
            return func(*args, **kwargs)
        
        return wrapper
    return decorator