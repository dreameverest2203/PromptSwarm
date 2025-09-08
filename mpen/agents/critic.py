"""
Critic Agent for evaluating prompts across multiple dimensions.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from .base import BaseAgent
from ..tasks.base import Task


class CriticAgent(BaseAgent):
    """
    Critic Agent specializes in evaluating prompt quality across multiple dimensions.
    
    Evaluates aspects like clarity, bias, safety, effectiveness, and domain-specific
    criteria. Can develop expertise in specific evaluation areas.
    """
    
    def __init__(
        self, 
        agent_id: str, 
        llm_config: Optional[Dict[str, Any]] = None,
        evaluation_dimensions: Optional[List[str]] = None
    ):
        """
        Initialize Critic Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            llm_config: Configuration for language model interface
            evaluation_dimensions: Specific dimensions to focus evaluation on
        """
        specialization_domains = [
            'bias_detection',
            'safety_evaluation',
            'clarity_assessment',
            'effectiveness_analysis',
            'logical_coherence',
            'instruction_quality'
        ]
        
        super().__init__(agent_id, llm_config, specialization_domains)
        
        self.evaluation_dimensions = evaluation_dimensions or [
            'clarity',
            'specificity',
            'bias',
            'safety',
            'effectiveness',
            'coherence',
            'completeness'
        ]
        
        # Dimension weights (can be learned over time)
        self.dimension_weights: Dict[str, float] = {
            dim: 1.0 / len(self.evaluation_dimensions)
            for dim in self.evaluation_dimensions
        }
        
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a prompt across multiple dimensions.
        
        Args:
            input_data: Contains 'prompt', 'task', and optional 'focus_dimensions'
            
        Returns:
            Evaluation scores and detailed feedback
        """
        prompt = input_data['prompt']
        task = input_data.get('task')
        focus_dimensions = input_data.get('focus_dimensions', self.evaluation_dimensions)
        
        # Perform evaluation
        evaluation_result = self.evaluate(prompt, task, focus_dimensions)
        
        # Record evaluation
        evaluation_record = {
            'prompt': prompt,
            'task_domain': getattr(task, 'domain', 'unknown') if task else 'unknown',
            'dimensions_evaluated': focus_dimensions,
            'scores': evaluation_result['dimension_scores'],
            'overall_score': evaluation_result['overall_score']
        }
        self.evaluation_history.append(evaluation_record)
        
        return evaluation_result
    
    def evaluate(
        self, 
        prompt: str, 
        task: Optional[Task] = None,
        focus_dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt across specified dimensions.
        
        Args:
            prompt: Prompt to evaluate
            task: Task context for evaluation
            focus_dimensions: Specific dimensions to focus on
            
        Returns:
            Dictionary with scores and feedback
        """
        if focus_dimensions is None:
            focus_dimensions = self.evaluation_dimensions
        
        dimension_scores = {}
        dimension_feedback = {}
        
        for dimension in focus_dimensions:
            score, feedback = self._evaluate_dimension(prompt, dimension, task)
            dimension_scores[dimension] = score
            dimension_feedback[dimension] = feedback
        
        # Calculate weighted overall score
        overall_score = sum(
            score * self.dimension_weights.get(dim, 1.0)
            for dim, score in dimension_scores.items()
        ) / len(dimension_scores)
        
        return {
            'overall_score': overall_score,
            'dimension_scores': dimension_scores,
            'dimension_feedback': dimension_feedback,
            'evaluator_id': self.agent_id,
            'confidence': self._calculate_confidence(dimension_scores)
        }
    
    def _evaluate_dimension(
        self, 
        prompt: str, 
        dimension: str, 
        task: Optional[Task] = None
    ) -> tuple[float, str]:
        """
        Evaluate a single dimension of the prompt.
        
        Args:
            prompt: Prompt to evaluate
            dimension: Dimension to evaluate
            task: Task context
            
        Returns:
            Tuple of (score, feedback)
        """
        if dimension == 'clarity':
            return self._evaluate_clarity(prompt)
        elif dimension == 'specificity':
            return self._evaluate_specificity(prompt)
        elif dimension == 'bias':
            return self._evaluate_bias(prompt)
        elif dimension == 'safety':
            return self._evaluate_safety(prompt)
        elif dimension == 'effectiveness':
            return self._evaluate_effectiveness(prompt, task)
        elif dimension == 'coherence':
            return self._evaluate_coherence(prompt)
        elif dimension == 'completeness':
            return self._evaluate_completeness(prompt, task)
        else:
            return 0.5, f"Unknown dimension: {dimension}"
    
    def _evaluate_clarity(self, prompt: str) -> tuple[float, str]:
        """Evaluate prompt clarity."""
        evaluation_prompt = f"""
        Evaluate the clarity of this prompt on a scale of 0-1:
        
        Prompt: "{prompt}"
        
        Consider:
        - Is the language clear and unambiguous?
        - Are instructions easy to understand?
        - Is the expected output format clear?
        
        Respond with just a number between 0 and 1, followed by a brief explanation.
        """
        
        response = self.llm.generate([
            {"role": "user", "content": evaluation_prompt}
        ])
        
        return self._parse_evaluation_response(response)
    
    def _evaluate_specificity(self, prompt: str) -> tuple[float, str]:
        """Evaluate prompt specificity."""
        evaluation_prompt = f"""
        Evaluate how specific and detailed this prompt is on a scale of 0-1:
        
        Prompt: "{prompt}"
        
        Consider:
        - Are the requirements clearly specified?
        - Is there enough detail to guide the response?
        - Are constraints and expectations explicit?
        
        Respond with just a number between 0 and 1, followed by a brief explanation.
        """
        
        response = self.llm.generate([
            {"role": "user", "content": evaluation_prompt}
        ])
        
        return self._parse_evaluation_response(response)
    
    def _evaluate_bias(self, prompt: str) -> tuple[float, str]:
        """Evaluate potential bias in the prompt (higher score = less bias)."""
        evaluation_prompt = f"""
        Evaluate this prompt for potential bias on a scale of 0-1 (1 = no bias, 0 = highly biased):
        
        Prompt: "{prompt}"
        
        Consider:
        - Does it contain cultural, gender, or demographic assumptions?
        - Are there leading questions or loaded language?
        - Does it promote stereotypes or unfair generalizations?
        
        Respond with just a number between 0 and 1, followed by a brief explanation.
        """
        
        response = self.llm.generate([
            {"role": "user", "content": evaluation_prompt}
        ])
        
        return self._parse_evaluation_response(response)
    
    def _evaluate_safety(self, prompt: str) -> tuple[float, str]:
        """Evaluate prompt safety (higher score = safer)."""
        evaluation_prompt = f"""
        Evaluate the safety of this prompt on a scale of 0-1 (1 = completely safe):
        
        Prompt: "{prompt}"
        
        Consider:
        - Could it lead to harmful or dangerous outputs?
        - Does it request inappropriate content?
        - Are there potential misuse scenarios?
        
        Respond with just a number between 0 and 1, followed by a brief explanation.
        """
        
        response = self.llm.generate([
            {"role": "user", "content": evaluation_prompt}
        ])
        
        return self._parse_evaluation_response(response)
    
    def _evaluate_effectiveness(
        self, 
        prompt: str, 
        task: Optional[Task] = None
    ) -> tuple[float, str]:
        """Evaluate how effective the prompt is for its intended task."""
        task_context = ""
        if task:
            task_context = f"\nTask context: {task.description}"
        
        evaluation_prompt = f"""
        Evaluate how effective this prompt would be at achieving its goal on a scale of 0-1:
        
        Prompt: "{prompt}"{task_context}
        
        Consider:
        - Does it clearly communicate the desired outcome?
        - Would it likely produce high-quality responses?
        - Is it well-structured for the intended task?
        
        Respond with just a number between 0 and 1, followed by a brief explanation.
        """
        
        response = self.llm.generate([
            {"role": "user", "content": evaluation_prompt}
        ])
        
        return self._parse_evaluation_response(response)
    
    def _evaluate_coherence(self, prompt: str) -> tuple[float, str]:
        """Evaluate logical coherence of the prompt."""
        evaluation_prompt = f"""
        Evaluate the logical coherence of this prompt on a scale of 0-1:
        
        Prompt: "{prompt}"
        
        Consider:
        - Do all parts of the prompt work together logically?
        - Are there contradictions or inconsistencies?
        - Does the structure make sense?
        
        Respond with just a number between 0 and 1, followed by a brief explanation.
        """
        
        response = self.llm.generate([
            {"role": "user", "content": evaluation_prompt}
        ])
        
        return self._parse_evaluation_response(response)
    
    def _evaluate_completeness(
        self, 
        prompt: str, 
        task: Optional[Task] = None
    ) -> tuple[float, str]:
        """Evaluate if the prompt is complete for its intended purpose."""
        task_context = ""
        if task:
            task_context = f"\nTask context: {task.description}"
        
        evaluation_prompt = f"""
        Evaluate how complete this prompt is on a scale of 0-1:
        
        Prompt: "{prompt}"{task_context}
        
        Consider:
        - Does it include all necessary information?
        - Are there missing instructions or context?
        - Would a user know exactly what to do?
        
        Respond with just a number between 0 and 1, followed by a brief explanation.
        """
        
        response = self.llm.generate([
            {"role": "user", "content": evaluation_prompt}
        ])
        
        return self._parse_evaluation_response(response)
    
    def _parse_evaluation_response(self, response: str) -> tuple[float, str]:
        """Parse the LLM evaluation response into score and feedback."""
        lines = response.strip().split('\n')
        
        try:
            # Try to extract the score from the first line
            first_line = lines[0].strip()
            
            # Look for a number between 0 and 1
            import re
            score_match = re.search(r'([0-1](?:\.\d+)?)', first_line)
            
            if score_match:
                score = float(score_match.group(1))
                # Get feedback from remaining lines
                feedback = '\n'.join(lines[1:]).strip() if len(lines) > 1 else first_line
            else:
                score = 0.5  # Default score if parsing fails
                feedback = response
            
            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))
            
        except (ValueError, IndexError):
            score = 0.5
            feedback = response
        
        return score, feedback
    
    def _calculate_confidence(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate confidence in the evaluation based on score consistency."""
        if not dimension_scores:
            return 0.0
        
        scores = list(dimension_scores.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Higher confidence when scores are consistent
        # Lower confidence when scores vary widely
        confidence = max(0.0, 1.0 - (std_score * 2))
        
        return confidence
    
    def update_dimension_weights(
        self, 
        dimension: str, 
        importance: float
    ) -> None:
        """Update the importance weight for an evaluation dimension."""
        if dimension in self.dimension_weights:
            self.dimension_weights[dimension] = max(0.0, min(2.0, importance))
            
            # Renormalize weights
            total_weight = sum(self.dimension_weights.values())
            if total_weight > 0:
                for dim in self.dimension_weights:
                    self.dimension_weights[dim] /= total_weight
    
    def _collaborate_impl(
        self, 
        other_agent: BaseAgent, 
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement collaboration with other agents."""
        if other_agent.agent_type == 'GeneratorAgent':
            # Evaluate multiple prompt variations
            variations = task_data.get('variations', [])
            evaluations = []
            
            for variation in variations:
                prompt = variation.get('prompt', '')
                evaluation = self.evaluate(prompt, task_data.get('task'))
                evaluations.append({
                    'prompt': prompt,
                    'evaluation': evaluation,
                    'generator': variation.get('generator')
                })
            
            # Rank variations by score
            evaluations.sort(key=lambda x: x['evaluation']['overall_score'], reverse=True)
            
            return {
                'success': True,
                'ranked_evaluations': evaluations,
                'collaboration_type': 'evaluate_variations'
            }
        
        return super()._collaborate_impl(other_agent, task_data)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation performance and patterns."""
        if not self.evaluation_history:
            return {'message': 'No evaluations performed yet'}
        
        # Calculate average scores by dimension
        dimension_averages = {}
        for dim in self.evaluation_dimensions:
            scores = [
                eval_record['scores'].get(dim, 0)
                for eval_record in self.evaluation_history
                if dim in eval_record['scores']
            ]
            if scores:
                dimension_averages[dim] = np.mean(scores)
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'dimension_averages': dimension_averages,
            'dimension_weights': self.dimension_weights.copy(),
            'evaluation_dimensions': self.evaluation_dimensions.copy()
        }
