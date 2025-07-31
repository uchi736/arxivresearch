"""
Progress tracking system for workflow execution

This module provides real-time progress tracking for the analysis workflow.
"""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class StepStatus(Enum):
    """Status of a workflow step"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class StepProgress:
    """Progress information for a single step"""
    name: str
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    current_item: Optional[str] = None
    total_items: int = 0
    completed_items: int = 0
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def progress_percentage(self) -> float:
        """Get progress percentage"""
        if self.total_items == 0:
            return 100.0 if self.status == StepStatus.COMPLETED else 0.0
        return (self.completed_items / self.total_items) * 100


class ProgressTracker:
    """Tracks progress of workflow execution"""
    
    WORKFLOW_STEPS = [
        "研究計画策定",
        "検索クエリ生成", 
        "論文検索",
        "論文翻訳",
        "フルテキスト処理",
        "論文分析",
        "レポート生成",
        "結果保存"
    ]
    
    def __init__(self):
        self.steps: Dict[str, StepProgress] = {}
        self.start_time = time.time()
        self.is_complete = False
        self.error_occurred = False
        
        # Initialize all steps
        for step_name in self.WORKFLOW_STEPS:
            self.steps[step_name] = StepProgress(name=step_name)
    
    def start_step(self, step_name: str, details: Optional[Dict[str, Any]] = None):
        """Start a workflow step"""
        if step_name in self.steps:
            step = self.steps[step_name]
            step.status = StepStatus.IN_PROGRESS
            step.start_time = time.time()
            if details:
                step.details.update(details)
                if "total_items" in details:
                    step.total_items = details["total_items"]
    
    def update_step(self, step_name: str, current_item: Optional[str] = None, 
                   details: Optional[Dict[str, Any]] = None):
        """Update progress of a step"""
        if step_name in self.steps:
            step = self.steps[step_name]
            if current_item:
                step.current_item = current_item
            if details:
                step.details.update(details)
                if "completed_items" in details:
                    step.completed_items = details["completed_items"]
                if "total_items" in details:
                    step.total_items = details["total_items"]
    
    def complete_step(self, step_name: str, details: Optional[Dict[str, Any]] = None):
        """Mark a step as completed"""
        if step_name in self.steps:
            step = self.steps[step_name]
            step.status = StepStatus.COMPLETED
            step.end_time = time.time()
            step.completed_items = step.total_items
            if details:
                step.details.update(details)
    
    def error_step(self, step_name: str, error_message: str):
        """Mark a step as errored"""
        if step_name in self.steps:
            step = self.steps[step_name]
            step.status = StepStatus.ERROR
            step.end_time = time.time()
            step.error_message = error_message
            self.error_occurred = True
    
    def skip_step(self, step_name: str):
        """Mark a step as skipped"""
        if step_name in self.steps:
            self.steps[step_name].status = StepStatus.SKIPPED
    
    @property
    def overall_progress(self) -> float:
        """Get overall progress percentage"""
        completed_steps = sum(1 for step in self.steps.values() 
                            if step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED])
        return (completed_steps / len(self.steps)) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time
    
    @property
    def current_step(self) -> Optional[StepProgress]:
        """Get the currently executing step"""
        for step in self.steps.values():
            if step.status == StepStatus.IN_PROGRESS:
                return step
        return None
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of the current status"""
        current = self.current_step
        return {
            "overall_progress": self.overall_progress,
            "elapsed_time": self.elapsed_time,
            "current_step": current.name if current else None,
            "current_item": current.current_item if current else None,
            "is_complete": self.is_complete,
            "error_occurred": self.error_occurred,
            "steps": {name: {
                "status": step.status.value,
                "duration": step.duration,
                "progress": step.progress_percentage,
                "details": step.details
            } for name, step in self.steps.items()}
        }
    
    def estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining time based on current progress"""
        if self.overall_progress == 0:
            return None
        
        # Simple linear estimation
        elapsed = self.elapsed_time
        if self.overall_progress > 0:
            total_estimated = elapsed / (self.overall_progress / 100)
            return max(0, total_estimated - elapsed)
        return None