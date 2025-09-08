"""
Mutation strategies for evolutionary prompt optimization.
"""

import random
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from ..agents.base import BaseAgent
from ..tasks.base import Task


class MutationStrategy(ABC):
    """Base class for mutation strategies."""
    
    @abstractmethod
    def mutate(
        self, 
        prompt: str, 
        agent: BaseAgent, 
        task: Optional[Task] = None
    ) -> str:
        """
        Mutate a prompt using the specified agent.
        
        Args:
            prompt: Original prompt to mutate
            agent: Agent to perform mutation
            task: Task context for mutation
            
        Returns:
            Mutated prompt
        """
        pass


class SemanticMutation(MutationStrategy):
    """Semantic mutation using agent capabilities."""
    
    def __init__(self, network, mutation_strength: float = 0.5):
        """
        Initialize semantic mutation.
        
        Args:
            network: Adaptive network for agent coordination
            mutation_strength: Strength of mutations (0-1)
        """
        self.network = network
        self.mutation_strength = mutation_strength
        
        self.mutation_types = [
            'paraphrase',
            'elaborate',
            'simplify',
            'restructure',
            'add_examples',
            'change_tone',
            'add_constraints',
            'remove_redundancy'
        ]
    
    def mutate(
        self, 
        prompt: str, 
        agent: BaseAgent, 
        task: Optional[Task] = None
    ) -> str:
        """Perform semantic mutation using agent capabilities."""
        # Select mutation type based on agent type and task
        mutation_type = self._select_mutation_type(agent, task)
        
        try:
            # Use agent to perform mutation
            if hasattr(agent, 'generate_variation'):
                # Generator agent
                return agent.generate_variation(prompt, mutation_type, task)
            else:
                # Other agent types - use general process method
                result = agent.process({
                    'prompt': prompt,
                    'task': task,
                    'operation': 'mutate',
                    'mutation_type': mutation_type,
                    'strength': self.mutation_strength
                })
                
                return result.get('variation', prompt)
                
        except Exception as e:
            # Fallback to simple mutation
            return self._simple_mutation(prompt, mutation_type)
    
    def _select_mutation_type(
        self, 
        agent: BaseAgent, 
        task: Optional[Task] = None
    ) -> str:
        """Select appropriate mutation type based on agent and task."""
        # Agent-specific preferences
        if agent.agent_type == 'GeneratorAgent':
            preferred_types = ['paraphrase', 'elaborate', 'restructure']
        elif agent.agent_type == 'CriticAgent':
            preferred_types = ['simplify', 'add_constraints', 'remove_redundancy']
        elif agent.agent_type == 'ValidatorAgent':
            preferred_types = ['add_examples', 'change_tone', 'elaborate']
        else:
            preferred_types = self.mutation_types
        
        # Task-specific adjustments
        if task and hasattr(task, 'domain'):
            if task.domain == 'mathematical_reasoning':
                preferred_types.extend(['add_examples', 'add_constraints'])
            elif task.domain == 'creative_writing':
                preferred_types.extend(['change_tone', 'elaborate'])
            elif task.domain == 'programming':
                preferred_types.extend(['add_examples', 'simplify'])
        
        return random.choice(preferred_types)
    
    def _simple_mutation(self, prompt: str, mutation_type: str) -> str:
        """Fallback simple mutation when agent-based mutation fails."""
        if mutation_type == 'paraphrase':
            return self._paraphrase_simple(prompt)
        elif mutation_type == 'elaborate':
            return self._elaborate_simple(prompt)
        elif mutation_type == 'simplify':
            return self._simplify_simple(prompt)
        elif mutation_type == 'add_examples':
            return self._add_examples_simple(prompt)
        else:
            # Default - just add variation text
            variations = [
                "Please consider this carefully: ",
                "Think step by step: ",
                "Be thorough in your response: ",
                "Consider multiple approaches: "
            ]
            return random.choice(variations) + prompt
    
    def _paraphrase_simple(self, prompt: str) -> str:
        """Simple paraphrasing by word substitution."""
        substitutions = {
            'solve': 'find the solution to',
            'explain': 'describe',
            'analyze': 'examine carefully',
            'create': 'develop',
            'write': 'compose',
            'think': 'consider'
        }
        
        result = prompt
        for original, replacement in substitutions.items():
            if original in result.lower():
                result = result.replace(original, replacement)
                break
        
        return result
    
    def _elaborate_simple(self, prompt: str) -> str:
        """Simple elaboration by adding detail."""
        elaborations = [
            " Provide detailed reasoning for your answer.",
            " Include specific examples to support your response.",
            " Consider multiple perspectives on this topic.",
            " Explain your thought process step by step."
        ]
        
        return prompt + random.choice(elaborations)
    
    def _simplify_simple(self, prompt: str) -> str:
        """Simple simplification by removing words."""
        # Remove common filler words
        fillers = ['please', 'kindly', 'very', 'really', 'quite', 'rather']
        
        words = prompt.split()
        filtered_words = [w for w in words if w.lower() not in fillers]
        
        return ' '.join(filtered_words) if filtered_words else prompt
    
    def _add_examples_simple(self, prompt: str) -> str:
        """Simple example addition."""
        example_text = " For example, consider similar problems and how they were approached."
        return prompt + example_text


class GeneticMutation(MutationStrategy):
    """Genetic-style mutation with crossover elements."""
    
    def __init__(self, mutation_rate: float = 0.1, crossover_templates: Optional[List[str]] = None):
        """
        Initialize genetic mutation.
        
        Args:
            mutation_rate: Probability of mutating each word
            crossover_templates: Templates for crossover operations
        """
        self.mutation_rate = mutation_rate
        self.crossover_templates = crossover_templates or [
            "Combine the following approaches: {prompt1} and {prompt2}",
            "Use elements from both: {prompt1} while also {prompt2}",
            "Start with {prompt1} then apply {prompt2}"
        ]
    
    def mutate(
        self, 
        prompt: str, 
        agent: BaseAgent, 
        task: Optional[Task] = None
    ) -> str:
        """Perform genetic-style mutation."""
        # Get other prompts from agent's history for crossover
        crossover_candidates = self._get_crossover_candidates(agent)
        
        if crossover_candidates and random.random() < 0.3:
            # Perform crossover mutation
            return self._crossover_mutate(prompt, crossover_candidates)
        else:
            # Perform point mutation
            return self._point_mutate(prompt)
    
    def _get_crossover_candidates(self, agent: BaseAgent) -> List[str]:
        """Get candidate prompts for crossover from agent's history."""
        candidates = []
        
        if hasattr(agent, 'generation_history'):
            # Generator agent has generation history
            for record in agent.generation_history[-5:]:  # Last 5
                if 'generated_variation' in record:
                    candidates.append(record['generated_variation'])
        
        return candidates
    
    def _crossover_mutate(self, prompt: str, candidates: List[str]) -> str:
        """Perform crossover mutation with candidate prompts."""
        candidate = random.choice(candidates)
        template = random.choice(self.crossover_templates)
        
        return template.format(prompt1=prompt, prompt2=candidate)
    
    def _point_mutate(self, prompt: str) -> str:
        """Perform point mutation on individual words."""
        words = prompt.split()
        mutated_words = []
        
        for word in words:
            if random.random() < self.mutation_rate:
                # Mutate this word
                mutated_word = self._mutate_word(word)
                mutated_words.append(mutated_word)
            else:
                mutated_words.append(word)
        
        return ' '.join(mutated_words)
    
    def _mutate_word(self, word: str) -> str:
        """Mutate a single word."""
        mutations = {
            'solve': 'resolve',
            'find': 'determine',
            'explain': 'clarify',
            'describe': 'outline',
            'analyze': 'evaluate',
            'create': 'generate',
            'write': 'draft',
            'calculate': 'compute',
            'simple': 'basic',
            'complex': 'complicated',
            'good': 'effective',
            'bad': 'poor'
        }
        
        return mutations.get(word.lower(), word)


class AdaptiveMutation(MutationStrategy):
    """Adaptive mutation that learns from success patterns."""
    
    def __init__(self, network):
        """
        Initialize adaptive mutation.
        
        Args:
            network: Adaptive network for learning patterns
        """
        self.network = network
        self.mutation_success_history: Dict[str, List[float]] = {}
        
        self.base_strategies = {
            'semantic': SemanticMutation(network),
            'genetic': GeneticMutation()
        }
    
    def mutate(
        self, 
        prompt: str, 
        agent: BaseAgent, 
        task: Optional[Task] = None
    ) -> str:
        """Perform adaptive mutation based on learned patterns."""
        # Select strategy based on historical success
        strategy_name = self._select_strategy(agent, task)
        strategy = self.base_strategies[strategy_name]
        
        # Perform mutation
        mutated_prompt = strategy.mutate(prompt, agent, task)
        
        # Record mutation attempt
        self._record_mutation_attempt(strategy_name, agent, task)
        
        return mutated_prompt
    
    def _select_strategy(self, agent: BaseAgent, task: Optional[Task] = None) -> str:
        """Select mutation strategy based on success history."""
        # Create key for this context
        context_key = f"{agent.agent_type}"
        if task and hasattr(task, 'domain'):
            context_key += f"_{task.domain}"
        
        # Get success rates for each strategy in this context
        strategy_scores = {}
        for strategy_name in self.base_strategies:
            full_key = f"{context_key}_{strategy_name}"
            if full_key in self.mutation_success_history:
                scores = self.mutation_success_history[full_key]
                strategy_scores[strategy_name] = sum(scores) / len(scores)
            else:
                strategy_scores[strategy_name] = 0.5  # Default
        
        # Select strategy with highest success rate (with some randomness)
        if random.random() < 0.2:  # 20% exploration
            return random.choice(list(self.base_strategies.keys()))
        else:
            return max(strategy_scores.items(), key=lambda x: x[1])[0]
    
    def _record_mutation_attempt(
        self, 
        strategy_name: str, 
        agent: BaseAgent, 
        task: Optional[Task] = None
    ) -> None:
        """Record mutation attempt for learning."""
        context_key = f"{agent.agent_type}"
        if task and hasattr(task, 'domain'):
            context_key += f"_{task.domain}"
        
        full_key = f"{context_key}_{strategy_name}"
        
        if full_key not in self.mutation_success_history:
            self.mutation_success_history[full_key] = []
        
        # For now, record neutral success (actual success will be updated later)
        self.mutation_success_history[full_key].append(0.5)
        
        # Keep limited history
        if len(self.mutation_success_history[full_key]) > 50:
            self.mutation_success_history[full_key] = self.mutation_success_history[full_key][-50:]
    
    def update_mutation_success(
        self,
        strategy_name: str,
        agent: BaseAgent,
        task: Optional[Task],
        success_score: float
    ) -> None:
        """Update success score for a mutation."""
        context_key = f"{agent.agent_type}"
        if task and hasattr(task, 'domain'):
            context_key += f"_{task.domain}"
        
        full_key = f"{context_key}_{strategy_name}"
        
        if full_key in self.mutation_success_history:
            # Update the most recent entry
            if self.mutation_success_history[full_key]:
                self.mutation_success_history[full_key][-1] = success_score


class HybridMutation(MutationStrategy):
    """Hybrid mutation combining multiple strategies."""
    
    def __init__(self, network, strategies: Optional[List[MutationStrategy]] = None):
        """
        Initialize hybrid mutation.
        
        Args:
            network: Adaptive network
            strategies: List of mutation strategies to combine
        """
        self.network = network
        self.strategies = strategies or [
            SemanticMutation(network),
            GeneticMutation()
        ]
        
        self.strategy_weights = [1.0] * len(self.strategies)
    
    def mutate(
        self, 
        prompt: str, 
        agent: BaseAgent, 
        task: Optional[Task] = None
    ) -> str:
        """Perform hybrid mutation using multiple strategies."""
        # Select strategy based on weights
        strategy_index = self._weighted_strategy_selection()
        strategy = self.strategies[strategy_index]
        
        # Perform mutation
        return strategy.mutate(prompt, agent, task)
    
    def _weighted_strategy_selection(self) -> int:
        """Select strategy based on weights."""
        total_weight = sum(self.strategy_weights)
        
        if total_weight == 0:
            return random.randint(0, len(self.strategies) - 1)
        
        # Normalize weights to probabilities
        probabilities = [w / total_weight for w in self.strategy_weights]
        
        # Select based on probabilities
        r = random.random()
        cumulative = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return i
        
        return len(self.strategies) - 1  # Fallback
    
    def update_strategy_weight(self, strategy_index: int, performance_score: float) -> None:
        """Update weight for a strategy based on performance."""
        if 0 <= strategy_index < len(self.strategy_weights):
            # Exponential moving average
            current_weight = self.strategy_weights[strategy_index]
            self.strategy_weights[strategy_index] = (
                0.8 * current_weight + 0.2 * performance_score
            )
            
            # Ensure minimum weight
            self.strategy_weights[strategy_index] = max(0.1, self.strategy_weights[strategy_index])
