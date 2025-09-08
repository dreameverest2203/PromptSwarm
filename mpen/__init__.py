"""
Meta-Prompt Evolutionary Networks (MPEN)

A collaborative multi-agent system for prompt optimization using adaptive networks
and evolutionary algorithms.
"""

from .system import MPENSystem
from .agents import GeneratorAgent, CriticAgent, ValidatorAgent, MetaAgent
from .network import AdaptiveNetwork
from .evolutionary import EvolutionaryOptimizer

__version__ = "0.1.0"
__all__ = [
    "MPENSystem",
    "GeneratorAgent", 
    "CriticAgent",
    "ValidatorAgent", 
    "MetaAgent",
    "AdaptiveNetwork",
    "EvolutionaryOptimizer"
]
