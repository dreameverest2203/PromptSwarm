"""
Generator Agent for creating prompt variations and mutations.
"""

import random
from typing import Dict, Any, List, Optional
import numpy as np

from .base import BaseAgent
from ..tasks.base import Task


class GeneratorAgent(BaseAgent):
    """
    Generator Agent specializes in creating diverse prompt variations.
    
    Uses various mutation strategies and can develop expertise in specific
    domains based on successful prompt generations.
    """
    
    def __init__(
        self, 
        agent_id: str, 
        llm_config: Optional[Dict[str, Any]] = None,
        mutation_strategies: Optional[List[str]] = None
    ):
        """
        Initialize Generator Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            llm_config: Configuration for language model interface
            mutation_strategies: List of mutation strategies to use
        """
        specialization_domains = [
            'mathematical_reasoning',
            'creative_writing', 
            'programming',
            'logical_reasoning',
            'instruction_following'
        ]
        
        super().__init__(agent_id, llm_config, specialization_domains)
        
        self.mutation_strategies = mutation_strategies or [
            'paraphrase',
            'add_examples', 
            'change_structure',
            'add_constraints',
            'simplify_language',
            'add_context',
            'change_perspective',
            'add_reasoning_steps'
        ]
        
        # Strategy effectiveness tracking
        self.strategy_performance: Dict[str, float] = {
            strategy: 0.5 for strategy in self.mutation_strategies
        }
        
        self.generation_history: List[Dict[str, Any]] = []
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate prompt variations based on input.
        
        Args:
            input_data: Contains 'base_prompt', 'task', and optional 'strategy'
            
        Returns:
            Generated prompt variations with metadata
        """
        base_prompt = input_data['base_prompt']
        task = input_data.get('task')
        requested_strategy = input_data.get('strategy')
        
        # Select mutation strategy
        if requested_strategy and requested_strategy in self.mutation_strategies:
            strategy = requested_strategy
        else:
            strategy = self._select_best_strategy(task)
        
        # Generate variation
        variation = self.generate_variation(base_prompt, strategy, task)
        
        # Record generation
        generation_record = {
            'base_prompt': base_prompt,
            'generated_variation': variation,
            'strategy_used': strategy,
            'task_domain': getattr(task, 'domain', 'unknown') if task else 'unknown'
        }
        self.generation_history.append(generation_record)
        
        return {
            'variation': variation,
            'strategy': strategy,
            'confidence': self.strategy_performance[strategy],
            'generator_id': self.agent_id
        }
    
    def generate_variation(
        self, 
        base_prompt: str, 
        strategy: Optional[str] = None,
        task: Optional[Task] = None
    ) -> str:
        """
        Generate a single prompt variation.
        
        Args:
            base_prompt: Original prompt to vary
            strategy: Specific strategy to use
            task: Task context for generation
            
        Returns:
            Generated prompt variation
        """
        if strategy is None:
            strategy = self._select_best_strategy(task)
        
        # Apply mutation strategy
        if strategy == 'paraphrase':
            return self._paraphrase_prompt(base_prompt)
        elif strategy == 'add_examples':
            return self._add_examples(base_prompt, task)
        elif strategy == 'change_structure':
            return self._change_structure(base_prompt)
        elif strategy == 'add_constraints':
            return self._add_constraints(base_prompt, task)
        elif strategy == 'simplify_language':
            return self._simplify_language(base_prompt)
        elif strategy == 'add_context':
            return self._add_context(base_prompt, task)
        elif strategy == 'change_perspective':
            return self._change_perspective(base_prompt)
        elif strategy == 'add_reasoning_steps':
            return self._add_reasoning_steps(base_prompt)
        else:
            return self._paraphrase_prompt(base_prompt)
    
    def _select_best_strategy(self, task: Optional[Task] = None) -> str:
        """Select the best mutation strategy based on performance history."""
        if task and hasattr(task, 'domain'):
            # Weight strategies by domain-specific performance
            domain_weights = self.domain_expertise.get(task.domain, 0.5)
            strategy_scores = {
                strategy: score * (1 + domain_weights)
                for strategy, score in self.strategy_performance.items()
            }
        else:
            strategy_scores = self.strategy_performance
        
        # Select strategy using softmax sampling
        strategies = list(strategy_scores.keys())
        scores = list(strategy_scores.values())
        
        # Convert to probabilities
        exp_scores = np.exp(np.array(scores) * 2)  # Temperature = 0.5
        probabilities = exp_scores / np.sum(exp_scores)
        
        return np.random.choice(strategies, p=probabilities)
    
    def _paraphrase_prompt(self, prompt: str) -> str:
        """Paraphrase the prompt while maintaining meaning."""
        system_prompt = """
        Paraphrase the following prompt while maintaining its core meaning and intent.
        Make it more natural and engaging while preserving all important instructions.
        """
        
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.strip()
    
    def _add_examples(self, prompt: str, task: Optional[Task] = None) -> str:
        """Add relevant examples to the prompt."""
        if task and hasattr(task, 'get_examples'):
            examples = task.get_examples(n=2)
            example_text = "\n\nExamples:\n" + "\n".join(examples)
        else:
            example_text = "\n\nFor example, think through each step carefully and show your reasoning."
        
        return prompt + example_text
    
    def _change_structure(self, prompt: str) -> str:
        """Change the structural organization of the prompt."""
        system_prompt = """
        Restructure the following prompt to improve clarity and flow.
        You can reorder sections, add numbering, use bullet points, or change formatting.
        Keep all the original content and instructions.
        """
        
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.strip()
    
    def _add_constraints(self, prompt: str, task: Optional[Task] = None) -> str:
        """Add helpful constraints or guidelines."""
        constraints = [
            "Be precise and accurate in your response.",
            "Show your step-by-step reasoning.",
            "Double-check your work before providing the final answer.",
            "Consider edge cases and potential errors."
        ]
        
        if task and hasattr(task, 'domain'):
            if task.domain == 'mathematical_reasoning':
                constraints.append("Verify your calculations at each step.")
            elif task.domain == 'creative_writing':
                constraints.append("Maintain consistency in tone and style.")
            elif task.domain == 'programming':
                constraints.append("Consider code efficiency and readability.")
        
        constraint_text = "\n\nGuidelines:\n" + "\n".join(f"- {c}" for c in constraints[:3])
        return prompt + constraint_text
    
    def _simplify_language(self, prompt: str) -> str:
        """Simplify the language while maintaining clarity."""
        system_prompt = """
        Rewrite the following prompt using simpler, clearer language.
        Remove unnecessary jargon and complex sentences while keeping all instructions.
        Make it more accessible and easier to understand.
        """
        
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.strip()
    
    def _add_context(self, prompt: str, task: Optional[Task] = None) -> str:
        """Add relevant context to help with the task."""
        context_additions = [
            "Think step by step and consider all aspects of the problem.",
            "Take your time to understand what is being asked.",
            "Consider multiple approaches before settling on your answer."
        ]
        
        if task and hasattr(task, 'domain'):
            if task.domain == 'mathematical_reasoning':
                context_additions.append("Remember to check your mathematical operations.")
            elif task.domain == 'creative_writing':
                context_additions.append("Consider your audience and the desired tone.")
        
        context = random.choice(context_additions)
        return f"{context}\n\n{prompt}"
    
    def _change_perspective(self, prompt: str) -> str:
        """Change the perspective or framing of the prompt."""
        perspectives = [
            "As an expert in this field, ",
            "Imagine you are teaching this to a student: ",
            "From a practical standpoint, ",
            "With careful attention to detail, "
        ]
        
        perspective = random.choice(perspectives)
        return perspective + prompt.lower()
    
    def _add_reasoning_steps(self, prompt: str) -> str:
        """Add explicit reasoning step instructions."""
        reasoning_addition = """
        
        Please approach this systematically:
        1. First, understand what is being asked
        2. Identify the key information and constraints
        3. Plan your approach
        4. Execute your solution step by step
        5. Verify your answer
        """
        
        return prompt + reasoning_addition
    
    def update_strategy_performance(
        self, 
        strategy: str, 
        performance_score: float
    ) -> None:
        """Update the performance score for a mutation strategy."""
        if strategy in self.strategy_performance:
            current_score = self.strategy_performance[strategy]
            # Exponential moving average
            self.strategy_performance[strategy] = (
                0.8 * current_score + 0.2 * performance_score
            )
    
    def _collaborate_impl(
        self, 
        other_agent: BaseAgent, 
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement collaboration with other agents."""
        if other_agent.agent_type == 'CriticAgent':
            # Generate multiple variations and let critic choose
            base_prompt = task_data.get('base_prompt', '')
            variations = []
            
            for _ in range(3):
                strategy = self._select_best_strategy(task_data.get('task'))
                variation = self.generate_variation(base_prompt, strategy)
                variations.append({
                    'prompt': variation,
                    'strategy': strategy,
                    'generator': self.agent_id
                })
            
            return {
                'success': True,
                'variations': variations,
                'collaboration_type': 'generate_for_critique'
            }
        
        return super()._collaborate_impl(other_agent, task_data)
    
    def get_strategy_summary(self) -> Dict[str, float]:
        """Get summary of strategy performance."""
        return self.strategy_performance.copy()
