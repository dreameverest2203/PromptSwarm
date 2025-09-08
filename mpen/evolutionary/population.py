"""
Population management for evolutionary prompt optimization.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import random


class PromptPopulation:
    """
    Manages a population of prompts with their fitness scores.
    
    Provides utilities for selection, statistics, and population management
    operations used in evolutionary optimization.
    """
    
    def __init__(self, prompts: List[str], fitness_scores: np.ndarray):
        """
        Initialize prompt population.
        
        Args:
            prompts: List of prompt strings
            fitness_scores: Numpy array of fitness scores
        """
        if len(prompts) != len(fitness_scores):
            raise ValueError("Prompts and fitness scores must have same length")
        
        self.prompts = prompts.copy()
        self.fitness_scores = fitness_scores.copy()
        self.size = len(prompts)
        
        # Sort by fitness (descending order)
        self._sort_by_fitness()
    
    def _sort_by_fitness(self) -> None:
        """Sort population by fitness scores in descending order."""
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        self.prompts = [self.prompts[i] for i in sorted_indices]
        self.fitness_scores = self.fitness_scores[sorted_indices]
    
    def get_best(self, n: int = 1) -> List[str]:
        """
        Get the n best prompts.
        
        Args:
            n: Number of best prompts to return
            
        Returns:
            List of best prompts
        """
        n = min(n, self.size)
        return self.prompts[:n]
    
    def get_worst(self, n: int = 1) -> List[str]:
        """
        Get the n worst prompts.
        
        Args:
            n: Number of worst prompts to return
            
        Returns:
            List of worst prompts
        """
        n = min(n, self.size)
        return self.prompts[-n:]
    
    def get_elite(self, n: int) -> List[str]:
        """
        Get elite individuals (same as get_best).
        
        Args:
            n: Number of elite individuals
            
        Returns:
            List of elite prompts
        """
        return self.get_best(n)
    
    def get_random_sample(self, n: int) -> List[str]:
        """
        Get random sample from population.
        
        Args:
            n: Number of prompts to sample
            
        Returns:
            List of randomly sampled prompts
        """
        n = min(n, self.size)
        return random.sample(self.prompts, n)
    
    def get_fitness_weighted_sample(self, n: int) -> List[str]:
        """
        Get sample weighted by fitness scores.
        
        Args:
            n: Number of prompts to sample
            
        Returns:
            List of fitness-weighted sampled prompts
        """
        if self.size == 0:
            return []
        
        n = min(n, self.size)
        
        # Normalize fitness scores to probabilities
        # Add small epsilon to handle negative or zero fitness
        adjusted_fitness = self.fitness_scores - np.min(self.fitness_scores) + 1e-6
        probabilities = adjusted_fitness / np.sum(adjusted_fitness)
        
        # Sample based on probabilities
        indices = np.random.choice(
            self.size, 
            size=n, 
            replace=False if n <= self.size else True,
            p=probabilities
        )
        
        return [self.prompts[i] for i in indices]
    
    def get_diverse_sample(self, n: int, diversity_threshold: float = 0.7) -> List[str]:
        """
        Get diverse sample by avoiding similar prompts.
        
        Args:
            n: Number of prompts to sample
            diversity_threshold: Minimum diversity score (0-1)
            
        Returns:
            List of diverse prompts
        """
        if self.size == 0 or n <= 0:
            return []
        
        if n >= self.size:
            return self.prompts.copy()
        
        selected = []
        candidates = self.prompts.copy()
        
        # Start with best prompt
        selected.append(candidates.pop(0))
        
        while len(selected) < n and candidates:
            best_candidate = None
            best_diversity = -1
            best_index = -1
            
            for i, candidate in enumerate(candidates):
                # Calculate minimum diversity with selected prompts
                min_diversity = min(
                    self._calculate_diversity(candidate, selected_prompt)
                    for selected_prompt in selected
                )
                
                if min_diversity > best_diversity:
                    best_diversity = min_diversity
                    best_candidate = candidate
                    best_index = i
            
            if best_candidate and best_diversity >= diversity_threshold:
                selected.append(best_candidate)
                candidates.pop(best_index)
            else:
                # If no diverse candidate found, take next best
                if candidates:
                    selected.append(candidates.pop(0))
        
        return selected
    
    def _calculate_diversity(self, prompt1: str, prompt2: str) -> float:
        """
        Calculate diversity score between two prompts.
        
        Args:
            prompt1: First prompt
            prompt2: Second prompt
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        # Simple diversity metric based on word overlap
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        
        if not words1 and not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        # Jaccard distance as diversity measure
        jaccard_similarity = intersection / union if union > 0 else 0.0
        diversity = 1.0 - jaccard_similarity
        
        return diversity
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get population statistics.
        
        Returns:
            Dictionary with population statistics
        """
        if self.size == 0:
            return {
                'size': 0,
                'min_fitness': 0.0,
                'max_fitness': 0.0,
                'mean_fitness': 0.0,
                'std_fitness': 0.0,
                'median_fitness': 0.0
            }
        
        return {
            'size': self.size,
            'min_fitness': float(np.min(self.fitness_scores)),
            'max_fitness': float(np.max(self.fitness_scores)),
            'mean_fitness': float(np.mean(self.fitness_scores)),
            'std_fitness': float(np.std(self.fitness_scores)),
            'median_fitness': float(np.median(self.fitness_scores))
        }
    
    def get_fitness_distribution(self, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get fitness distribution histogram.
        
        Args:
            bins: Number of histogram bins
            
        Returns:
            Tuple of (counts, bin_edges)
        """
        if self.size == 0:
            return np.array([]), np.array([])
        
        counts, bin_edges = np.histogram(self.fitness_scores, bins=bins)
        return counts, bin_edges
    
    def get_prompt_by_rank(self, rank: int) -> Optional[str]:
        """
        Get prompt by fitness rank.
        
        Args:
            rank: Fitness rank (0 = best, 1 = second best, etc.)
            
        Returns:
            Prompt at specified rank or None if rank is invalid
        """
        if 0 <= rank < self.size:
            return self.prompts[rank]
        return None
    
    def get_fitness_by_rank(self, rank: int) -> Optional[float]:
        """
        Get fitness score by rank.
        
        Args:
            rank: Fitness rank (0 = best, 1 = second best, etc.)
            
        Returns:
            Fitness score at specified rank or None if rank is invalid
        """
        if 0 <= rank < self.size:
            return float(self.fitness_scores[rank])
        return None
    
    def find_prompt_rank(self, prompt: str) -> Optional[int]:
        """
        Find the rank of a specific prompt.
        
        Args:
            prompt: Prompt to find rank for
            
        Returns:
            Rank of the prompt or None if not found
        """
        try:
            return self.prompts.index(prompt)
        except ValueError:
            return None
    
    def update_fitness(self, prompt: str, new_fitness: float) -> bool:
        """
        Update fitness score for a specific prompt.
        
        Args:
            prompt: Prompt to update
            new_fitness: New fitness score
            
        Returns:
            True if update successful, False if prompt not found
        """
        try:
            index = self.prompts.index(prompt)
            self.fitness_scores[index] = new_fitness
            self._sort_by_fitness()  # Re-sort after update
            return True
        except ValueError:
            return False
    
    def add_prompt(self, prompt: str, fitness: float) -> None:
        """
        Add a new prompt to the population.
        
        Args:
            prompt: New prompt to add
            fitness: Fitness score for the new prompt
        """
        self.prompts.append(prompt)
        self.fitness_scores = np.append(self.fitness_scores, fitness)
        self.size += 1
        self._sort_by_fitness()
    
    def remove_worst(self, n: int = 1) -> List[str]:
        """
        Remove n worst prompts from population.
        
        Args:
            n: Number of worst prompts to remove
            
        Returns:
            List of removed prompts
        """
        n = min(n, self.size)
        if n <= 0:
            return []
        
        removed_prompts = self.prompts[-n:]
        self.prompts = self.prompts[:-n]
        self.fitness_scores = self.fitness_scores[:-n]
        self.size -= n
        
        return removed_prompts
    
    def trim_to_size(self, target_size: int) -> List[str]:
        """
        Trim population to target size by removing worst individuals.
        
        Args:
            target_size: Desired population size
            
        Returns:
            List of removed prompts
        """
        if target_size >= self.size:
            return []
        
        n_to_remove = self.size - target_size
        return self.remove_worst(n_to_remove)
    
    def get_diversity_metrics(self) -> Dict[str, float]:
        """
        Calculate diversity metrics for the population.
        
        Returns:
            Dictionary with diversity metrics
        """
        if self.size < 2:
            return {
                'mean_pairwise_diversity': 0.0,
                'min_pairwise_diversity': 0.0,
                'max_pairwise_diversity': 0.0
            }
        
        diversities = []
        
        for i in range(self.size):
            for j in range(i + 1, self.size):
                diversity = self._calculate_diversity(self.prompts[i], self.prompts[j])
                diversities.append(diversity)
        
        diversities = np.array(diversities)
        
        return {
            'mean_pairwise_diversity': float(np.mean(diversities)),
            'min_pairwise_diversity': float(np.min(diversities)),
            'max_pairwise_diversity': float(np.max(diversities)),
            'std_pairwise_diversity': float(np.std(diversities))
        }
    
    def clone(self) -> 'PromptPopulation':
        """
        Create a deep copy of the population.
        
        Returns:
            New PromptPopulation instance
        """
        return PromptPopulation(self.prompts.copy(), self.fitness_scores.copy())
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, index: int) -> Tuple[str, float]:
        """Get prompt and fitness by index."""
        if 0 <= index < self.size:
            return self.prompts[index], float(self.fitness_scores[index])
        raise IndexError("Population index out of range")
    
    def __iter__(self):
        """Iterate over (prompt, fitness) pairs."""
        for i in range(self.size):
            yield self.prompts[i], float(self.fitness_scores[i])
    
    def __repr__(self) -> str:
        return f"PromptPopulation(size={self.size}, best_fitness={self.fitness_scores[0]:.3f})"
