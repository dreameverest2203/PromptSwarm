"""
Agent implementations for the MPEN system.

Contains four types of specialized agents:
- GeneratorAgent: Creates prompt variations and mutations
- CriticAgent: Evaluates prompts across multiple dimensions
- ValidatorAgent: Tests prompts on unseen tasks and edge cases
- MetaAgent: Coordinates the network and manages global optimization
"""

from .base import BaseAgent
from .generator import GeneratorAgent
from .critic import CriticAgent
from .validator import ValidatorAgent
from .meta import MetaAgent

__all__ = [
    "BaseAgent",
    "GeneratorAgent", 
    "CriticAgent",
    "ValidatorAgent", 
    "MetaAgent"
]
