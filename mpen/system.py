"""
Main MPEN System that orchestrates the multi-agent prompt optimization process.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

from .agents import GeneratorAgent, CriticAgent, ValidatorAgent, MetaAgent
from .network import AdaptiveNetwork
from .evolutionary import EvolutionaryOptimizer
from .tasks.base import Task
from .utils.logging import setup_logger


@dataclass
class OptimizationResult:
    """Results from a prompt optimization run."""
    best_prompt: str
    best_score: float
    iteration_history: List[Dict[str, Any]]
    network_evolution: Dict[str, Any]
    agent_contributions: Dict[str, float]


class MPENSystem:
    """
    Meta-Prompt Evolutionary Networks System
    
    Coordinates multiple specialized agents to collaboratively optimize prompts
    using adaptive network connections and evolutionary algorithms.
    """
    
    def __init__(
        self,
        num_generators: int = 3,
        num_critics: int = 2,
        num_validators: int = 2,
        num_meta_agents: int = 1,
        network_adaptation_rate: float = 0.1,
        evolution_params: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        logger_name: str = "mpen"
    ):
        """
        Initialize the MPEN system.
        
        Args:
            num_generators: Number of generator agents
            num_critics: Number of critic agents
            num_validators: Number of validator agents
            num_meta_agents: Number of meta agents
            network_adaptation_rate: Rate at which network connections adapt
            evolution_params: Parameters for evolutionary optimization
            llm_config: Configuration for language models
            logger_name: Name for the logger
        """
        self.logger = setup_logger(logger_name)
        
        # Initialize agents
        self.generators = [
            GeneratorAgent(f"gen_{i}", llm_config) 
            for i in range(num_generators)
        ]
        self.critics = [
            CriticAgent(f"critic_{i}", llm_config)
            for i in range(num_critics)
        ]
        self.validators = [
            ValidatorAgent(f"val_{i}", llm_config)
            for i in range(num_validators)
        ]
        self.meta_agents = [
            MetaAgent(f"meta_{i}", llm_config)
            for i in range(num_meta_agents)
        ]
        
        # Collect all agents
        self.all_agents = (
            self.generators + self.critics + 
            self.validators + self.meta_agents
        )
        
        # Initialize network and optimizer
        self.network = AdaptiveNetwork(
            agents=self.all_agents,
            adaptation_rate=network_adaptation_rate
        )
        
        evolution_params = evolution_params or {}
        self.optimizer = EvolutionaryOptimizer(
            network=self.network,
            **evolution_params
        )
        
        self.logger.info(
            f"Initialized MPEN system with {len(self.all_agents)} agents"
        )
    
    def optimize(
        self,
        initial_prompt: str,
        task: Task,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-4,
        population_size: int = 20
    ) -> OptimizationResult:
        """
        Optimize a prompt using the MPEN system.
        
        Args:
            initial_prompt: Starting prompt to optimize
            task: Task to optimize for
            max_iterations: Maximum optimization iterations
            convergence_threshold: Threshold for convergence detection
            population_size: Size of prompt population
            
        Returns:
            OptimizationResult with best prompt and optimization history
        """
        self.logger.info(f"Starting optimization for task: {task.name}")
        
        # Initialize prompt population
        population = self._initialize_population(
            initial_prompt, population_size
        )
        
        iteration_history = []
        best_score = float('-inf')
        best_prompt = initial_prompt
        
        for iteration in range(max_iterations):
            self.logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            
            # Evaluate current population
            scores = self._evaluate_population(population, task)
            
            # Update best if improved
            current_best_idx = np.argmax(scores)
            current_best_score = scores[current_best_idx]
            
            if current_best_score > best_score:
                best_score = current_best_score
                best_prompt = population[current_best_idx]
                self.logger.info(f"New best score: {best_score:.4f}")
            
            # Record iteration data
            iteration_data = {
                'iteration': iteration,
                'best_score': best_score,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'network_state': self.network.get_state_summary()
            }
            iteration_history.append(iteration_data)
            
            # Check convergence
            if self._check_convergence(iteration_history, convergence_threshold):
                self.logger.info(f"Converged at iteration {iteration}")
                break
            
            # Evolve population using network
            population = self.optimizer.evolve_population(
                population, scores, task
            )
            
            # Update network connections based on agent performance
            self.network.update_connections(iteration_data)
        
        # Compile results
        result = OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            iteration_history=iteration_history,
            network_evolution=self.network.get_evolution_history(),
            agent_contributions=self._calculate_agent_contributions()
        )
        
        self.logger.info(f"Optimization complete. Best score: {best_score:.4f}")
        return result
    
    def _initialize_population(
        self, initial_prompt: str, population_size: int
    ) -> List[str]:
        """Initialize the prompt population using generator agents."""
        population = [initial_prompt]
        
        # Generate variations using generator agents
        for _ in range(population_size - 1):
            generator = np.random.choice(self.generators)
            variation = generator.generate_variation(
                initial_prompt, 
                context={"population_size": len(population)}
            )
            population.append(variation)
        
        return population
    
    def _evaluate_population(
        self, population: List[str], task: Task
    ) -> np.ndarray:
        """Evaluate population using critic and validator agents."""
        scores = []
        
        for prompt in population:
            # Get task-specific score
            task_score = task.evaluate(prompt)
            
            # Get critic evaluations
            critic_scores = []
            for critic in self.critics:
                critic_score = critic.evaluate(prompt, task)
                critic_scores.append(critic_score)
            
            # Get validator evaluations
            validator_scores = []
            for validator in self.validators:
                validator_score = validator.validate(prompt, task)
                validator_scores.append(validator_score)
            
            # Combine scores (weighted average)
            combined_score = (
                0.5 * task_score +
                0.3 * np.mean(critic_scores) +
                0.2 * np.mean(validator_scores)
            )
            scores.append(combined_score)
        
        return np.array(scores)
    
    def _check_convergence(
        self, history: List[Dict[str, Any]], threshold: float
    ) -> bool:
        """Check if optimization has converged."""
        if len(history) < 5:
            return False
        
        recent_scores = [h['best_score'] for h in history[-5:]]
        score_variance = np.var(recent_scores)
        
        return score_variance < threshold
    
    def _calculate_agent_contributions(self) -> Dict[str, float]:
        """Calculate how much each agent contributed to the optimization."""
        contributions = {}
        
        for agent in self.all_agents:
            # This would be based on actual tracking during optimization
            # For now, return placeholder values
            contributions[agent.agent_id] = np.random.random()
        
        return contributions
    
    def get_network_visualization(self) -> Dict[str, Any]:
        """Get data for visualizing the agent network."""
        return self.network.get_visualization_data()
    
    def save_state(self, filepath: str) -> None:
        """Save the current system state."""
        # Implementation for saving system state
        pass
    
    def load_state(self, filepath: str) -> None:
        """Load a previously saved system state."""
        # Implementation for loading system state
        pass
