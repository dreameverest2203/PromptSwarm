"""
Base task class for MPEN evaluation tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass


@dataclass
class TaskResult:
    """Result from task evaluation."""
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class Task(ABC):
    """
    Base class for all MPEN evaluation tasks.
    
    Tasks define specific evaluation criteria and provide methods
    for testing prompt effectiveness in different domains.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        domain: str,
        difficulty: float = 0.5,
        max_execution_time: float = 30.0
    ):
        """
        Initialize base task.
        
        Args:
            name: Task name
            description: Task description
            domain: Task domain (e.g., 'mathematical_reasoning')
            difficulty: Task difficulty (0.0 to 1.0)
            max_execution_time: Maximum execution time in seconds
        """
        self.name = name
        self.description = description
        self.domain = domain
        self.difficulty = difficulty
        self.max_execution_time = max_execution_time
        
        # Evaluation tracking
        self.evaluation_history: List[Dict[str, Any]] = []
        self.performance_stats: Dict[str, float] = {}
        
        self.logger = logging.getLogger(f"mpen.task.{name}")
        
    @abstractmethod
    def evaluate(self, prompt: str, **kwargs) -> float:
        """
        Evaluate a prompt on this task.
        
        Args:
            prompt: Prompt to evaluate
            **kwargs: Additional evaluation parameters
            
        Returns:
            Score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def get_test_cases(self) -> List[Dict[str, Any]]:
        """
        Get test cases for this task.
        
        Returns:
            List of test case dictionaries
        """
        pass
    
    def evaluate_detailed(self, prompt: str, **kwargs) -> TaskResult:
        """
        Perform detailed evaluation with full results.
        
        Args:
            prompt: Prompt to evaluate
            **kwargs: Additional evaluation parameters
            
        Returns:
            TaskResult object with detailed information
        """
        import time
        
        start_time = time.time()
        
        try:
            score = self.evaluate(prompt, **kwargs)
            execution_time = time.time() - start_time
            
            # Record evaluation
            self._record_evaluation(prompt, score, execution_time)
            
            return TaskResult(
                score=score,
                details=self._get_evaluation_details(prompt, score),
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            self.logger.error(f"Task evaluation failed: {e}")
            
            return TaskResult(
                score=0.0,
                details={'error': str(e)},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _record_evaluation(self, prompt: str, score: float, execution_time: float) -> None:
        """Record evaluation for analysis."""
        evaluation_record = {
            'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
            'score': score,
            'execution_time': execution_time,
            'timestamp': time.time()
        }
        
        self.evaluation_history.append(evaluation_record)
        
        # Keep limited history
        if len(self.evaluation_history) > 1000:
            self.evaluation_history = self.evaluation_history[-1000:]
        
        # Update performance stats
        self._update_performance_stats()
    
    def _update_performance_stats(self) -> None:
        """Update performance statistics."""
        if not self.evaluation_history:
            return
        
        scores = [e['score'] for e in self.evaluation_history]
        times = [e['execution_time'] for e in self.evaluation_history]
        
        self.performance_stats = {
            'mean_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'mean_execution_time': sum(times) / len(times),
            'total_evaluations': len(self.evaluation_history)
        }
    
    def _get_evaluation_details(self, prompt: str, score: float) -> Dict[str, Any]:
        """Get detailed evaluation information."""
        return {
            'prompt_length': len(prompt),
            'score': score,
            'task_name': self.name,
            'task_domain': self.domain,
            'difficulty': self.difficulty
        }
    
    def get_examples(self, n: int = 3) -> List[str]:
        """
        Get example inputs/outputs for this task.
        
        Args:
            n: Number of examples to return
            
        Returns:
            List of example strings
        """
        test_cases = self.get_test_cases()
        
        examples = []
        for case in test_cases[:n]:
            if 'input' in case and 'expected_output' in case:
                example = f"Input: {case['input']}\nExpected: {case['expected_output']}"
                examples.append(example)
        
        return examples
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this task."""
        return {
            'task_info': {
                'name': self.name,
                'description': self.description,
                'domain': self.domain,
                'difficulty': self.difficulty
            },
            'performance_stats': self.performance_stats.copy(),
            'evaluation_count': len(self.evaluation_history)
        }
    
    def benchmark_prompt(
        self, 
        prompt: str, 
        num_runs: int = 5
    ) -> Dict[str, float]:
        """
        Benchmark a prompt with multiple runs.
        
        Args:
            prompt: Prompt to benchmark
            num_runs: Number of evaluation runs
            
        Returns:
            Benchmark statistics
        """
        scores = []
        times = []
        
        for _ in range(num_runs):
            result = self.evaluate_detailed(prompt)
            scores.append(result.score)
            times.append(result.execution_time)
        
        return {
            'mean_score': sum(scores) / len(scores),
            'std_score': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
            'min_score': min(scores),
            'max_score': max(scores),
            'mean_time': sum(times) / len(times),
            'total_runs': num_runs
        }
    
    def compare_prompts(
        self, 
        prompts: List[str], 
        num_runs: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple prompts on this task.
        
        Args:
            prompts: List of prompts to compare
            num_runs: Number of runs per prompt
            
        Returns:
            Comparison results
        """
        results = {}
        
        for i, prompt in enumerate(prompts):
            prompt_id = f"prompt_{i+1}"
            results[prompt_id] = self.benchmark_prompt(prompt, num_runs)
            results[prompt_id]['prompt'] = prompt[:50] + '...' if len(prompt) > 50 else prompt
        
        return results
    
    def get_difficulty_adjusted_score(self, raw_score: float) -> float:
        """
        Adjust score based on task difficulty.
        
        Args:
            raw_score: Raw evaluation score
            
        Returns:
            Difficulty-adjusted score
        """
        # Higher difficulty tasks get bonus for same raw score
        difficulty_bonus = 1.0 + (self.difficulty - 0.5) * 0.2
        adjusted_score = raw_score * difficulty_bonus
        
        return min(1.0, max(0.0, adjusted_score))
    
    def validate_prompt_format(self, prompt: str) -> Tuple[bool, Optional[str]]:
        """
        Validate prompt format for this task.
        
        Args:
            prompt: Prompt to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not prompt or not prompt.strip():
            return False, "Prompt cannot be empty"
        
        if len(prompt) > 10000:
            return False, "Prompt too long (max 10000 characters)"
        
        return True, None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', domain='{self.domain}')"
