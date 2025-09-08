"""
Experiment framework for MPEN research.
"""

from .config import (
    ExperimentConfig,
    SmallScaleConfig, 
    MediumScaleConfig,
    LargeScaleConfig,
    BenchmarkConfig,
    DomainConfig,
    LLMConfig,
    ExperimentSuite,
    EXPERIMENT_CONFIGS,
    DOMAIN_CONFIGS,
    LLM_CONFIGS
)

__all__ = [
    "ExperimentConfig",
    "SmallScaleConfig",
    "MediumScaleConfig", 
    "LargeScaleConfig",
    "BenchmarkConfig",
    "DomainConfig",
    "LLMConfig",
    "ExperimentSuite",
    "EXPERIMENT_CONFIGS",
    "DOMAIN_CONFIGS",
    "LLM_CONFIGS"
]
