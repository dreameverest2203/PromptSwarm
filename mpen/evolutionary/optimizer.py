"""
Main evolutionary optimizer for the MPEN system.
"""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .population import PromptPopulation
from .selection import TournamentSelection, SelectionStrategy
from .mutation import SemanticMutation, MutationStrategy
from ..network.adaptive_network import AdaptiveNetwork
from ..tasks.base import Task


class EvolutionaryOptimizer:
    """
    Evolutionary optimizer that coordinates agent networks to evolve prompt populations.
    
    Uses network-based collaboration between agents to guide evolutionary operations
    like selection, mutation, and crossover.
    """
    
    def __init__(
        self,
        network: AdaptiveNetwork,
        population_size: int = 20,
        elite_ratio: float = 0.2,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.6,
        selection_strategy: Optional[SelectionStrategy] = None,
        mutation_strategy: Optional[MutationStrategy] = None
    ):
        """
        Initialize the evolutionary optimizer.
        
        Args:
            network: Adaptive network of agents
            population_size: Size of prompt population
            elite_ratio: Ratio of elite individuals to preserve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            selection_strategy: Strategy for parent selection
            mutation_strategy: Strategy for mutations
        """
        self.network = network
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Initialize strategies
        self.selection_strategy = selection_strategy or TournamentSelection(tournament_size=3)
        self.mutation_strategy = mutation_strategy or SemanticMutation(network)
        
        # Evolution tracking
        self.generation = 0
        self.evolution_history: List[Dict[str, Any]] = []
        self.best_fitness_history: List[float] = []
        
        self.logger = logging.getLogger("mpen.evolutionary")
        self.logger.info("Initialized evolutionary optimizer")
    
    def evolve_population(
        self,
        current_population: List[str],
        fitness_scores: np.ndarray,
        task: Task
    ) -> List[str]:
        """
        Evolve the population for one generation.
        
        Args:
            current_population: Current prompt population
            fitness_scores: Fitness scores for each prompt
            task: Task being optimized for
            
        Returns:
            New evolved population
        """
        self.generation += 1
        self.logger.info(f"Evolution generation {self.generation}")
        
        # Create population object
        population = PromptPopulation(current_population, fitness_scores)
        
        # Record generation statistics
        self._record_generation_stats(population, task)
        
        # Evolve using network-guided operations
        new_population = self._evolve_with_network(population, task)
        
        return new_population.prompts
    
    def _evolve_with_network(
        self,
        population: PromptPopulation,
        task: Task
    ) -> PromptPopulation:
        """Evolve population using network-guided operations."""
        
        # 1. Elite preservation
        elite_size = int(self.population_size * self.elite_ratio)
        elite_prompts = population.get_elite(elite_size)
        
        # 2. Network-guided selection and reproduction
        new_prompts = list(elite_prompts)
        
        while len(new_prompts) < self.population_size:
            # Use network to decide on operation
            operation = self._select_evolutionary_operation(task)
            
            if operation == 'crossover' and random.random() < self.crossover_rate:
                # Network-guided crossover
                offspring = self._network_guided_crossover(population, task)
                if offspring:
                    new_prompts.extend(offspring)
            
            elif operation == 'mutation' and random.random() < self.mutation_rate:
                # Network-guided mutation
                mutated = self._network_guided_mutation(population, task)
                if mutated:
                    new_prompts.append(mutated)
            
            else:
                # Selection-based reproduction
                parent = self.selection_strategy.select(population)
                new_prompts.append(parent)
        
        # Trim to exact population size
        new_prompts = new_prompts[:self.population_size]
        
        # Create new population (fitness will be evaluated externally)
        return PromptPopulation(new_prompts, np.zeros(len(new_prompts)))
    
    def _select_evolutionary_operation(self, task: Task) -> str:
        """Select evolutionary operation based on network state and task."""
        # Get network statistics
        network_stats = self.network.get_network_statistics()
        
        # Decision logic based on network connectivity and performance
        avg_strength = network_stats['strength_stats']['mean']
        active_connections = network_stats['active_connections']
        
        if avg_strength > 0.7 and active_connections > 5:
            # Strong network - favor collaborative crossover
            return 'crossover'
        elif avg_strength < 0.3:
            # Weak network - favor exploration through mutation
            return 'mutation'
        else:
            # Balanced - random choice
            return random.choice(['crossover', 'mutation', 'selection'])
    
    def _network_guided_crossover(
        self,
        population: PromptPopulation,
        task: Task
    ) -> Optional[List[str]]:
        """Perform crossover guided by agent network collaborations."""
        # Find collaborative agent pairs
        collaborative_pairs = self._get_collaborative_pairs()
        
        if not collaborative_pairs:
            return None
        
        # Select a collaborative pair
        agent1_id, agent2_id = random.choice(collaborative_pairs)
        
        # Get agents
        agent1 = self.network.agents.get(agent1_id)
        agent2 = self.network.agents.get(agent2_id)
        
        if not agent1 or not agent2:
            return None
        
        # Select parents using different agents
        parent1 = self.selection_strategy.select(population)
        parent2 = self.selection_strategy.select(population)
        
        # Perform collaborative crossover
        offspring = self._collaborative_crossover(
            parent1, parent2, agent1, agent2, task
        )
        
        # Record collaboration
        if offspring:
            self.network.record_interaction(
                agent1_id, agent2_id,
                success=True,
                benefit=0.7,  # Assume moderate benefit
                domain=getattr(task, 'domain', None)
            )
        
        return offspring
    
    def _network_guided_mutation(
        self,
        population: PromptPopulation,
        task: Task
    ) -> Optional[str]:
        """Perform mutation guided by specialist agents."""
        # Find specialist agents for the task domain
        specialists = self._get_domain_specialists(task)
        
        if not specialists:
            # Fallback to random agent
            specialists = list(self.network.agents.keys())
        
        if not specialists:
            return None
        
        # Select specialist agent
        specialist_id = random.choice(specialists)
        specialist = self.network.agents.get(specialist_id)
        
        if not specialist:
            return None
        
        # Select parent
        parent = self.selection_strategy.select(population)
        
        # Perform specialist mutation
        mutated = self.mutation_strategy.mutate(
            parent, specialist, task
        )
        
        return mutated
    
    def _get_collaborative_pairs(self) -> List[Tuple[str, str]]:
        """Get pairs of agents with strong collaborative connections."""
        pairs = []
        
        for connection in self.network.connections.values():
            if (connection.is_active and 
                connection.connection_type.value == 'collaborative' and
                connection.strength > 0.6):
                pairs.append((connection.source_agent_id, connection.target_agent_id))
        
        return pairs
    
    def _get_domain_specialists(self, task: Task) -> List[str]:
        """Get agents that specialize in the task domain."""
        if not hasattr(task, 'domain'):
            return []
        
        domain = task.domain
        specialists = []
        
        for agent_id, agent in self.network.agents.items():
            if domain in agent.domain_expertise:
                if agent.domain_expertise[domain] > 0.6:
                    specialists.append(agent_id)
        
        return specialists
    
    def _collaborative_crossover(
        self,
        parent1: str,
        parent2: str,
        agent1,
        agent2,
        task: Task
    ) -> Optional[List[str]]:
        """Perform crossover with collaboration between two agents."""
        try:
            # Agent 1 processes first parent
            result1 = agent1.process({
                'prompt': parent1,
                'task': task,
                'operation': 'crossover_prepare',
                'partner_prompt': parent2
            })
            
            # Agent 2 processes second parent  
            result2 = agent2.process({
                'prompt': parent2,
                'task': task,
                'operation': 'crossover_prepare',
                'partner_prompt': parent1
            })
            
            # Combine results to create offspring
            if 'variation' in result1 and 'variation' in result2:
                offspring1 = self._combine_variations(
                    result1['variation'], result2['variation'], bias=0.7
                )
                offspring2 = self._combine_variations(
                    result2['variation'], result1['variation'], bias=0.7
                )
                return [offspring1, offspring2]
            
        except Exception as e:
            self.logger.error(f"Collaborative crossover failed: {e}")
        
        return None
    
    def _combine_variations(
        self,
        variation1: str,
        variation2: str,
        bias: float = 0.5
    ) -> str:
        """Combine two prompt variations with specified bias."""
        # Simple combination strategy - could be made more sophisticated
        parts1 = variation1.split('.')
        parts2 = variation2.split('.')
        
        combined_parts = []
        max_parts = max(len(parts1), len(parts2))
        
        for i in range(max_parts):
            if random.random() < bias:
                if i < len(parts1):
                    combined_parts.append(parts1[i])
            else:
                if i < len(parts2):
                    combined_parts.append(parts2[i])
        
        return '. '.join(combined_parts).strip()
    
    def _record_generation_stats(
        self,
        population: PromptPopulation,
        task: Task
    ) -> None:
        """Record statistics for the current generation."""
        stats = population.get_statistics()
        
        generation_record = {
            'generation': self.generation,
            'population_size': len(population.prompts),
            'best_fitness': stats['max_fitness'],
            'mean_fitness': stats['mean_fitness'],
            'fitness_std': stats['std_fitness'],
            'task_domain': getattr(task, 'domain', 'unknown'),
            'network_connections': len(self.network.connections),
            'active_connections': sum(
                1 for c in self.network.connections.values() if c.is_active
            )
        }
        
        self.evolution_history.append(generation_record)
        self.best_fitness_history.append(stats['max_fitness'])
        
        # Keep limited history
        if len(self.evolution_history) > 1000:
            self.evolution_history = self.evolution_history[-1000:]
            self.best_fitness_history = self.best_fitness_history[-1000:]
    
    def get_convergence_metrics(self) -> Dict[str, float]:
        """Calculate convergence metrics for the evolutionary process."""
        if len(self.best_fitness_history) < 10:
            return {
                'convergence_rate': 0.0,
                'fitness_improvement': 0.0,
                'generations_since_improvement': 0
            }
        
        recent_fitness = self.best_fitness_history[-10:]
        
        # Calculate convergence rate (variance in recent fitness)
        convergence_rate = 1.0 - np.var(recent_fitness)
        
        # Calculate fitness improvement over last 10 generations
        fitness_improvement = recent_fitness[-1] - recent_fitness[0]
        
        # Count generations since last improvement
        generations_since_improvement = 0
        best_recent = max(recent_fitness)
        
        for i in range(len(recent_fitness) - 1, -1, -1):
            if recent_fitness[i] < best_recent:
                generations_since_improvement += 1
            else:
                break
        
        return {
            'convergence_rate': max(0.0, min(1.0, convergence_rate)),
            'fitness_improvement': fitness_improvement,
            'generations_since_improvement': generations_since_improvement
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization process."""
        if not self.evolution_history:
            return {'message': 'No evolution history available'}
        
        convergence_metrics = self.get_convergence_metrics()
        
        return {
            'total_generations': self.generation,
            'best_fitness_achieved': max(self.best_fitness_history) if self.best_fitness_history else 0,
            'current_fitness': self.best_fitness_history[-1] if self.best_fitness_history else 0,
            'convergence_metrics': convergence_metrics,
            'population_size': self.population_size,
            'evolution_parameters': {
                'elite_ratio': self.elite_ratio,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            },
            'network_utilization': {
                'total_agents': len(self.network.agents),
                'active_connections': sum(
                    1 for c in self.network.connections.values() if c.is_active
                )
            }
        }
    
    def adapt_parameters(self, performance_metrics: Dict[str, float]) -> None:
        """Adapt evolutionary parameters based on performance."""
        convergence_metrics = self.get_convergence_metrics()
        
        # Increase mutation rate if converging too quickly
        if convergence_metrics['convergence_rate'] > 0.9:
            self.mutation_rate = min(0.8, self.mutation_rate * 1.1)
            self.logger.info(f"Increased mutation rate to {self.mutation_rate:.3f}")
        
        # Decrease mutation rate if not converging
        elif convergence_metrics['generations_since_improvement'] > 10:
            self.mutation_rate = max(0.1, self.mutation_rate * 0.9)
            self.logger.info(f"Decreased mutation rate to {self.mutation_rate:.3f}")
        
        # Adjust crossover rate based on network connectivity
        network_stats = self.network.get_network_statistics()
        avg_strength = network_stats['strength_stats']['mean']
        
        if avg_strength > 0.7:
            self.crossover_rate = min(0.9, self.crossover_rate * 1.05)
        elif avg_strength < 0.3:
            self.crossover_rate = max(0.2, self.crossover_rate * 0.95)
    
    def reset(self) -> None:
        """Reset the optimizer state."""
        self.generation = 0
        self.evolution_history.clear()
        self.best_fitness_history.clear()
        self.logger.info("Reset evolutionary optimizer")
