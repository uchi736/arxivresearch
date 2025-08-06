"""
Base utilities and common imports for workflow nodes
"""

import time
from typing import Dict, Optional
from src.core.models import AdvancedAgentState, PaperMemory
from src.core.progress_tracker import ProgressTracker, StepStatus
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Paper memories storage - shared across nodes for a single workflow execution
# Use a function to get/reset the storage to avoid global state issues
_paper_memories: Dict[str, PaperMemory] = {}

def get_paper_memories() -> Dict[str, PaperMemory]:
    """Get the paper memories storage"""
    global _paper_memories
    return _paper_memories

def clear_paper_memories():
    """Clear the paper memories storage - should be called at workflow start"""
    global _paper_memories
    logger.debug(f"Clearing paper memories (had {len(_paper_memories)} entries)")
    _paper_memories.clear()

# For backward compatibility
paper_memories = get_paper_memories()


def get_progress_tracker(state: AdvancedAgentState) -> ProgressTracker:
    """Get or create progress tracker from state"""
    if state.get("progress_tracker") is None:
        tracker = ProgressTracker()
    else:
        # Reconstruct from dict
        tracker = ProgressTracker()
        tracker_data = state["progress_tracker"]
        tracker.start_time = tracker_data.get("start_time", tracker.start_time)
        tracker.is_complete = tracker_data.get("is_complete", False)
        tracker.error_occurred = tracker_data.get("error_occurred", False)
        # Restore step states
        for step_name, step_data in tracker_data.get("steps", {}).items():
            if step_name in tracker.steps:
                step = tracker.steps[step_name]
                step.status = StepStatus(step_data.get("status", "pending"))
                step.start_time = step_data.get("start_time")
                step.end_time = step_data.get("end_time")
                step.details = step_data.get("details", {})
    return tracker


def save_progress_tracker(tracker: ProgressTracker) -> Dict:
    """Convert tracker to dict for state storage"""
    return {
        "start_time": tracker.start_time,
        "is_complete": tracker.is_complete,
        "error_occurred": tracker.error_occurred,
        "steps": {
            name: {
                "status": step.status.value,
                "start_time": step.start_time,
                "end_time": step.end_time,
                "details": step.details
            } for name, step in tracker.steps.items()
        }
    }