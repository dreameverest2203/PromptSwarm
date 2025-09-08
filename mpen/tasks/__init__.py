"""
Task definitions and evaluation framework for MPEN.
"""

from .base import Task
from .math_reasoning import MathReasoningTask
from .creative_writing import CreativeWritingTask  
from .programming import ProgrammingTask
from .task_factory import TaskFactory

__all__ = [
    "Task",
    "MathReasoningTask",
    "CreativeWritingTask", 
    "ProgrammingTask",
    "TaskFactory"
]
