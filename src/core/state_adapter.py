"""
State Adapter for LangGraph Integration

This module provides adapters to integrate the improved state management
system with existing LangGraph workflows.
"""

from typing import Dict, Any, Callable, Optional
from functools import wraps
from src.core.state_manager import StateManager, WorkflowState, StateValidationError
from src.core.models import AdvancedAgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StateAdapter:
    """
    Adapter to bridge between LangGraph state and improved state management
    """
    
    def __init__(self):
        """Initialize the adapter"""
        self._state_managers: Dict[str, StateManager] = {}
    
    def get_or_create_manager(self, workflow_id: str) -> StateManager:
        """
        Get or create a state manager for a workflow
        
        Args:
            workflow_id: Unique workflow identifier
            
        Returns:
            StateManager instance
        """
        if workflow_id not in self._state_managers:
            self._state_managers[workflow_id] = StateManager()
            logger.debug(f"Created new state manager for workflow: {workflow_id}")
        
        return self._state_managers[workflow_id]
    
    def wrap_node(self, node_func: Callable, node_name: str) -> Callable:
        """
        Wrap a node function with state management
        
        Args:
            node_func: Original node function
            node_name: Name of the node for validation
            
        Returns:
            Wrapped function with state management
        """
        @wraps(node_func)
        def wrapped(langgraph_state: AdvancedAgentState, container: Any) -> Dict[str, Any]:
            # Extract workflow ID (use initial_query as simple ID for now)
            workflow_id = str(langgraph_state.get("initial_query", "default"))
            
            # Get state manager
            manager = self.get_or_create_manager(workflow_id)
            
            # Import current state from LangGraph
            if manager.current_state is None:
                # Initialize if first time
                manager.import_from_langgraph(langgraph_state)
            else:
                # Update with any changes from LangGraph
                updates = {}
                current_dict = manager.export_for_langgraph()
                
                for key, value in langgraph_state.items():
                    if key in current_dict and current_dict[key] != value:
                        updates[key] = value
                
                if updates:
                    manager.batch_update(updates)
            
            # Validate transition
            try:
                manager.validate_transition(node_name)
            except StateValidationError as e:
                logger.error(f"State validation failed for {node_name}: {e}")
                # Continue anyway for backward compatibility
            
            # Execute original node
            result = node_func(langgraph_state, container)
            
            # Update state manager with results
            if isinstance(result, dict):
                try:
                    manager.batch_update(result)
                except Exception as e:
                    logger.error(f"Failed to update state after {node_name}: {e}")
            
            # Return result for LangGraph
            return result
        
        return wrapped
    
    def get_state_snapshot(self, workflow_id: str) -> Optional[WorkflowState]:
        """
        Get current state snapshot for a workflow
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Current WorkflowState or None
        """
        if workflow_id in self._state_managers:
            return self._state_managers[workflow_id].current_state
        return None
    
    def clear_workflow(self, workflow_id: str):
        """
        Clear state manager for a workflow
        
        Args:
            workflow_id: Workflow identifier
        """
        if workflow_id in self._state_managers:
            del self._state_managers[workflow_id]
            logger.debug(f"Cleared state manager for workflow: {workflow_id}")


# Global adapter instance
_adapter = StateAdapter()


def adapt_node_with_state_management(node_name: str):
    """
    Decorator to add state management to a node
    
    Args:
        node_name: Name of the node for validation
        
    Example:
        @adapt_node_with_state_management("search_papers")
        def search_papers_node(state, container):
            # node implementation
            return {"found_papers": papers}
    """
    def decorator(func):
        return _adapter.wrap_node(func, node_name)
    
    return decorator


def get_workflow_state(workflow_id: str) -> Optional[WorkflowState]:
    """
    Get current state for a workflow
    
    Args:
        workflow_id: Workflow identifier
        
    Returns:
        Current WorkflowState or None
    """
    return _adapter.get_state_snapshot(workflow_id)


def clear_workflow_state(workflow_id: str):
    """
    Clear state for a workflow
    
    Args:
        workflow_id: Workflow identifier
    """
    _adapter.clear_workflow(workflow_id)


# Validation helpers for common patterns
def validate_papers_available(state: WorkflowState, min_papers: int = 1) -> bool:
    """Check if enough papers are available"""
    return len(state.found_papers) >= min_papers


def validate_analysis_complete(state: WorkflowState) -> bool:
    """Check if analysis is complete"""
    return len(state.analyzed_papers) > 0 and all(
        paper.get("analysis") is not None 
        for paper in state.analyzed_papers
    )


def validate_report_ready(state: WorkflowState) -> bool:
    """Check if report can be generated"""
    return validate_analysis_complete(state) and state.final_report is None


# State update helpers
def safe_append_to_list(state_dict: Dict[str, Any], key: str, item: Any) -> Dict[str, Any]:
    """
    Safely append item to a list in state
    
    Args:
        state_dict: Current state dictionary
        key: Key of the list field
        item: Item to append
        
    Returns:
        Updated state dictionary
    """
    current_list = state_dict.get(key, [])
    new_list = current_list + [item]
    return {key: new_list}


def safe_extend_list(state_dict: Dict[str, Any], key: str, items: list) -> Dict[str, Any]:
    """
    Safely extend a list in state
    
    Args:
        state_dict: Current state dictionary  
        key: Key of the list field
        items: Items to extend with
        
    Returns:
        Updated state dictionary
    """
    current_list = state_dict.get(key, [])
    new_list = current_list + items
    return {key: new_list}