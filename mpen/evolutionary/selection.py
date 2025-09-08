"""
Selection strategies for evolutionary optimization.
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

from .population import PromptPopulation


class SelectionStrategy(ABC):
    """Base class for selection strategies."""
    
    @abstractmethod
    def select(self, population: PromptPopulation, n: int = 1) -> str:
        """
        Select individual(s) from population.
        
        Args:
            population: Population to select from
            n: Number of individuals to select
            
        Returns:
            Selected prompt string
        """
        pass


class TournamentSelection(SelectionStrategy):
    """Tournament selection strategy."""
    
    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Size of tournament
        """
        self.tournament_size = tournament_size
    
    def select(self, population: PromptPopulation, n: int = 1) -> str:
        """Select individual using tournament selection."""
        if population.size == 0:
            raise ValueError("Cannot select from empty population")
        
        # Select tournament participants
        tournament_size = min(self.tournament_size, population.size)
        participants = random.sample(range(population.size), tournament_size)
        
        # Find best participant
        best_index = participants[0]
        best_fitness = population.fitness_scores[best_index]
        
        for index in participants[1:]:
            if population.fitness_scores[index] > best_fitness:
                best_fitness = population.fitness_scores[index]
                best_index = index
        
        return population.prompts[best_index]


class RouletteSelection(SelectionStrategy):
    """Roulette wheel selection strategy."""
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize roulette selection.
        
        Args:
            temperature: Temperature parameter for softmax
        """
        self.temperature = temperature
    
    def select(self, population: PromptPopulation, n: int = 1) -> str:
        """Select individual using roulette wheel selection."""
        if population.size == 0:
            raise ValueError("Cannot select from empty population")
        
        # Handle negative fitness scores
        fitness_scores = population.fitness_scores.copy()
        min_fitness = np.min(fitness_scores)
        if min_fitness < 0:
            fitness_scores = fitness_scores - min_fitness + 1e-6
        
        # Apply temperature
        if self.temperature != 1.0:
            fitness_scores = fitness_scores / self.temperature
        
        # Convert to probabilities
        probabilities = fitness_scores / np.sum(fitness_scores)
        
        # Select based on probabilities
        selected_index = np.random.choice(population.size, p=probabilities)
        
        return population.prompts[selected_index]


class RankSelection(SelectionStrategy):
    """Rank-based selection strategy."""
    
    def __init__(self, selection_pressure: float = 1.5):
        """
        Initialize rank selection.
        
        Args:
            selection_pressure: Selection pressure parameter (1.0-2.0)
        """
        self.selection_pressure = max(1.0, min(2.0, selection_pressure))
    
    def select(self, population: PromptPopulation, n: int = 1) -> str:
        """Select individual using rank-based selection."""
        if population.size == 0:
            raise ValueError("Cannot select from empty population")
        
        # Calculate rank-based probabilities
        ranks = np.arange(population.size, 0, -1)  # Best = highest rank
        
        # Linear ranking formula
        probabilities = (
            (2 - self.selection_pressure) / population.size +
            (2 * ranks * (self.selection_pressure - 1)) / 
            (population.size * (population.size - 1))
        )
        
        # Normalize probabilities
        probabilities = probabilities / np.sum(probabilities)
        
        # Select based on probabilities
        selected_index = np.random.choice(population.size, p=probabilities)
        
        return population.prompts[selected_index]


class ElitistSelection(SelectionStrategy):
    """Elitist selection - always selects from top performers."""
    
    def __init__(self, elite_ratio: float = 0.2):
        """
        Initialize elitist selection.
        
        Args:
            elite_ratio: Ratio of population considered elite
        """
        self.elite_ratio = max(0.1, min(1.0, elite_ratio))
    
    def select(self, population: PromptPopulation, n: int = 1) -> str:
        """Select individual from elite group."""
        if population.size == 0:
            raise ValueError("Cannot select from empty population")
        
        # Calculate elite size
        elite_size = max(1, int(population.size * self.elite_ratio))
        
        # Select randomly from elite
        elite_index = random.randint(0, elite_size - 1)
        
        return population.prompts[elite_index]


class DiversitySelection(SelectionStrategy):
    """Diversity-based selection to maintain population diversity."""
    
    def __init__(self, diversity_weight: float = 0.3):
        """
        Initialize diversity selection.
        
        Args:
            diversity_weight: Weight of diversity vs fitness (0-1)
        """
        self.diversity_weight = max(0.0, min(1.0, diversity_weight))
        self.fitness_weight = 1.0 - self.diversity_weight
    
    def select(self, population: PromptPopulation, n: int = 1) -> str:
        """Select individual balancing fitness and diversity."""
        if population.size == 0:
            raise ValueError("Cannot select from empty population")
        
        if population.size == 1:
            return population.prompts[0]
        
        # Calculate diversity scores for each individual
        diversity_scores = []
        
        for i in range(population.size):
            # Calculate average diversity with rest of population
            diversities = []
            for j in range(population.size):
                if i != j:
                    diversity = population._calculate_diversity(
                        population.prompts[i], population.prompts[j]
                    )
                    diversities.append(diversity)
            
            avg_diversity = np.mean(diversities) if diversities else 0.0
            diversity_scores.append(avg_diversity)
        
        diversity_scores = np.array(diversity_scores)
        
        # Normalize fitness and diversity scores
        fitness_scores = population.fitness_scores.copy()
        
        # Handle negative scores
        if np.min(fitness_scores) < 0:
            fitness_scores = fitness_scores - np.min(fitness_scores)
        
        # Normalize to [0, 1]
        if np.max(fitness_scores) > 0:
            fitness_scores = fitness_scores / np.max(fitness_scores)
        
        if np.max(diversity_scores) > 0:
            diversity_scores = diversity_scores / np.max(diversity_scores)
        
        # Combine fitness and diversity
        combined_scores = (
            self.fitness_weight * fitness_scores +
            self.diversity_weight * diversity_scores
        )
        
        # Select based on combined scores
        probabilities = combined_scores / np.sum(combined_scores)
        selected_index = np.random.choice(population.size, p=probabilities)
        
        return population.prompts[selected_index]


class AdaptiveSelection(SelectionStrategy):
    """Adaptive selection that changes strategy based on population state."""
    
    def __init__(self):
        """Initialize adaptive selection with multiple strategies."""
        self.strategies = {
            'tournament': TournamentSelection(tournament_size=3),
            'roulette': RouletteSelection(temperature=1.0),
            'rank': RankSelection(selection_pressure=1.5),
            'elitist': ElitistSelection(elite_ratio=0.2),
            'diversity': DiversitySelection(diversity_weight=0.3)
        }
        
        self.strategy_performance = {name: 0.5 for name in self.strategies}
        self.last_selections = []
    
    def select(self, population: PromptPopulation, n: int = 1) -> str:
        """Select using adaptive strategy choice."""
        if population.size == 0:
            raise ValueError("Cannot select from empty population")
        
        # Choose strategy based on population characteristics
        strategy_name = self._choose_strategy(population)
        strategy = self.strategies[strategy_name]
        
        # Perform selection
        selected = strategy.select(population, n)
        
        # Record selection for performance tracking
        self.last_selections.append({
            'strategy': strategy_name,
            'selected': selected,
            'population_size': population.size
        })
        
        # Keep limited history
        if len(self.last_selections) > 100:
            self.last_selections = self.last_selections[-100:]
        
        return selected
    
    def _choose_strategy(self, population: PromptPopulation) -> str:
        """Choose selection strategy based on population state."""
        stats = population.get_statistics()
        diversity_metrics = population.get_diversity_metrics()
        
        # Decision logic based on population characteristics
        fitness_variance = stats['std_fitness']
        mean_diversity = diversity_metrics['mean_pairwise_diversity']
        
        if fitness_variance < 0.1:
            # Low fitness variance - use diversity selection
            return 'diversity'
        elif mean_diversity < 0.3:
            # Low diversity - use diversity selection
            return 'diversity'
        elif stats['mean_fitness'] > 0.8:
            # High fitness - use elitist selection
            return 'elitist'
        elif population.size < 10:
            # Small population - use tournament
            return 'tournament'
        else:
            # Default - use roulette
            return 'roulette'
    
    def update_performance(
        self, 
        strategy_name: str, 
        performance_score: float
    ) -> None:
        """Update performance score for a strategy."""
        if strategy_name in self.strategy_performance:
            current_score = self.strategy_performance[strategy_name]
            # Exponential moving average
            self.strategy_performance[strategy_name] = (
                0.8 * current_score + 0.2 * performance_score
            )
