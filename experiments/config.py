"""
Experiment configurations for MPEN research.
"""

from typing import Dict, Any, List
import os


class ExperimentConfig:
    """Base experiment configuration."""
    
    def __init__(self, name: str):
        self.name = name
        self.base_config = {
            'max_iterations': 50,
            'population_size': 20,
            'num_generators': 3,
            'num_critics': 2,
            'num_validators': 2,
            'num_meta_agents': 1,
            'convergence_threshold': 1e-4
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        return self.base_config.copy()


class SmallScaleConfig(ExperimentConfig):
    """Configuration for small-scale experiments."""
    
    def __init__(self):
        super().__init__("small_scale")
        self.base_config.update({
            'max_iterations': 20,
            'population_size': 10,
            'num_generators': 2,
            'num_critics': 1,
            'num_validators': 1,
            'num_meta_agents': 1
        })


class MediumScaleConfig(ExperimentConfig):
    """Configuration for medium-scale experiments."""
    
    def __init__(self):
        super().__init__("medium_scale")
        self.base_config.update({
            'max_iterations': 50,
            'population_size': 20,
            'num_generators': 3,
            'num_critics': 2,
            'num_validators': 2,
            'num_meta_agents': 1
        })


class LargeScaleConfig(ExperimentConfig):
    """Configuration for large-scale experiments."""
    
    def __init__(self):
        super().__init__("large_scale")
        self.base_config.update({
            'max_iterations': 100,
            'population_size': 40,
            'num_generators': 5,
            'num_critics': 3,
            'num_validators': 3,
            'num_meta_agents': 2
        })


class BenchmarkConfig:
    """Benchmark experiment configurations."""
    
    @staticmethod
    def get_baseline_config() -> Dict[str, Any]:
        """Get baseline single-agent configuration for comparison."""
        return {
            'name': 'baseline_single_agent',
            'max_iterations': 50,
            'population_size': 20,
            'num_generators': 1,
            'num_critics': 0,
            'num_validators': 0,
            'num_meta_agents': 0,
            'network_adaptation_rate': 0.0  # No network adaptation
        }
    
    @staticmethod
    def get_multi_agent_configs() -> List[Dict[str, Any]]:
        """Get different multi-agent configurations for comparison."""
        return [
            {
                'name': 'cooperative_only',
                'max_iterations': 50,
                'population_size': 20,
                'num_generators': 2,
                'num_critics': 2,
                'num_validators': 1,
                'num_meta_agents': 1,
                'cooperation_bias': 1.0,
                'competition_bias': 0.0
            },
            {
                'name': 'competitive_only',
                'max_iterations': 50,
                'population_size': 20,
                'num_generators': 2,
                'num_critics': 2,
                'num_validators': 1,
                'num_meta_agents': 1,
                'cooperation_bias': 0.0,
                'competition_bias': 1.0
            },
            {
                'name': 'balanced_cooperation_competition',
                'max_iterations': 50,
                'population_size': 20,
                'num_generators': 3,
                'num_critics': 2,
                'num_validators': 2,
                'num_meta_agents': 1,
                'cooperation_bias': 0.5,
                'competition_bias': 0.5
            }
        ]


class DomainConfig:
    """Domain-specific experiment configurations."""
    
    @staticmethod
    def get_math_config() -> Dict[str, Any]:
        """Configuration optimized for mathematical reasoning."""
        return {
            'task_type': 'math_reasoning',
            'task_config': {
                'difficulty': 0.7,
                'name': 'Mathematical Reasoning Benchmark'
            },
            'initial_prompts': [
                "Solve this mathematical problem step by step, showing all work:",
                "Find the solution using clear mathematical reasoning:",
                "Calculate the answer with detailed explanations:",
                "Work through this problem systematically:"
            ],
            'evaluation_focus': ['correctness', 'reasoning_quality', 'step_clarity']
        }
    
    @staticmethod
    def get_creative_writing_config() -> Dict[str, Any]:
        """Configuration optimized for creative writing."""
        return {
            'task_type': 'creative_writing',
            'task_config': {
                'difficulty': 0.6,
                'name': 'Creative Writing Benchmark'
            },
            'initial_prompts': [
                "Write a creative and engaging story:",
                "Create an original piece of writing with vivid imagery:",
                "Compose a compelling narrative with strong characters:",
                "Develop an imaginative story with rich descriptions:"
            ],
            'evaluation_focus': ['creativity', 'engagement', 'language_quality']
        }
    
    @staticmethod
    def get_programming_config() -> Dict[str, Any]:
        """Configuration optimized for programming tasks."""
        return {
            'task_type': 'programming',
            'task_config': {
                'difficulty': 0.8,
                'name': 'Programming Benchmark'
            },
            'initial_prompts': [
                "Write clean, efficient code to solve this problem:",
                "Implement a solution with proper documentation:",
                "Create a well-structured program that solves:",
                "Develop an algorithm with clear comments:"
            ],
            'evaluation_focus': ['correctness', 'efficiency', 'code_quality']
        }


class LLMConfig:
    """LLM provider configurations."""
    
    @staticmethod
    def get_openai_config() -> Dict[str, Any]:
        """OpenAI GPT configuration."""
        return {
            'provider': 'openai',
            'model': 'gpt-3.5-turbo',
            'api_key': os.getenv('OPENAI_API_KEY'),
            'default_params': {
                'temperature': 0.7,
                'max_tokens': 1000
            }
        }
    
    @staticmethod
    def get_anthropic_config() -> Dict[str, Any]:
        """Anthropic Claude configuration."""
        return {
            'provider': 'anthropic',
            'model': 'claude-3-sonnet-20240229',
            'api_key': os.getenv('ANTHROPIC_API_KEY'),
            'default_params': {
                'max_tokens': 1000
            }
        }
    
    @staticmethod
    def get_mock_config() -> Dict[str, Any]:
        """Mock LLM configuration for testing."""
        return {
            'provider': 'mock',
            'mock_responses': [
                "I'll solve this step by step with clear reasoning and detailed explanations.",
                "Let me approach this systematically, breaking down the problem into manageable parts.",
                "Here's my comprehensive solution with thorough analysis and proper structure.",
                "I'll work through this methodically, showing all steps and providing clear rationale.",
                "Let me provide a detailed response with examples and careful consideration of all aspects."
            ]
        }


class ExperimentSuite:
    """Complete experiment suite configurations."""
    
    @staticmethod
    def get_paper_experiments() -> List[Dict[str, Any]]:
        """Get experiments for research paper."""
        return [
            {
                'name': 'baseline_comparison',
                'description': 'Compare MPEN against single-agent baseline',
                'configs': [
                    BenchmarkConfig.get_baseline_config(),
                    MediumScaleConfig().get_config()
                ],
                'domains': ['math', 'writing', 'programming'],
                'repetitions': 5
            },
            {
                'name': 'cooperation_vs_competition',
                'description': 'Compare cooperative vs competitive strategies',
                'configs': BenchmarkConfig.get_multi_agent_configs(),
                'domains': ['math', 'writing', 'programming'],
                'repetitions': 3
            },
            {
                'name': 'scaling_analysis',
                'description': 'Analyze performance scaling with system size',
                'configs': [
                    SmallScaleConfig().get_config(),
                    MediumScaleConfig().get_config(),
                    LargeScaleConfig().get_config()
                ],
                'domains': ['math'],
                'repetitions': 3
            },
            {
                'name': 'domain_specialization',
                'description': 'Study agent specialization across domains',
                'configs': [MediumScaleConfig().get_config()],
                'domains': ['math', 'writing', 'programming'],
                'repetitions': 5,
                'analyze_specialization': True
            }
        ]
    
    @staticmethod
    def get_ablation_studies() -> List[Dict[str, Any]]:
        """Get ablation study configurations."""
        base_config = MediumScaleConfig().get_config()
        
        return [
            {
                'name': 'no_critics',
                'description': 'Remove critic agents',
                'config': {**base_config, 'num_critics': 0}
            },
            {
                'name': 'no_validators',
                'description': 'Remove validator agents',
                'config': {**base_config, 'num_validators': 0}
            },
            {
                'name': 'no_meta_agents',
                'description': 'Remove meta agents',
                'config': {**base_config, 'num_meta_agents': 0}
            },
            {
                'name': 'no_network_adaptation',
                'description': 'Disable network adaptation',
                'config': {**base_config, 'network_adaptation_rate': 0.0}
            },
            {
                'name': 'high_mutation_rate',
                'description': 'Increase mutation rate',
                'config': {**base_config, 'mutation_rate': 0.6}
            },
            {
                'name': 'low_mutation_rate',
                'description': 'Decrease mutation rate',
                'config': {**base_config, 'mutation_rate': 0.1}
            }
        ]


# Predefined experiment configurations
EXPERIMENT_CONFIGS = {
    'small': SmallScaleConfig(),
    'medium': MediumScaleConfig(),
    'large': LargeScaleConfig(),
}

DOMAIN_CONFIGS = {
    'math': DomainConfig.get_math_config(),
    'writing': DomainConfig.get_creative_writing_config(),
    'programming': DomainConfig.get_programming_config()
}

LLM_CONFIGS = {
    'openai': LLMConfig.get_openai_config(),
    'anthropic': LLMConfig.get_anthropic_config(),
    'mock': LLMConfig.get_mock_config()
}
