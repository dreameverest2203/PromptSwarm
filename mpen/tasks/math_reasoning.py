"""
Mathematical reasoning task for MPEN evaluation.
"""

import re
import random
from typing import Dict, Any, List, Optional
import numpy as np

from .base import Task
from ..utils.llm_interface import LLMInterface


class MathReasoningTask(Task):
    """
    Mathematical reasoning task that evaluates prompts on mathematical problem solving.
    
    Tests various aspects of mathematical reasoning including:
    - Arithmetic operations
    - Algebraic manipulation
    - Word problems
    - Multi-step reasoning
    - Problem decomposition
    """
    
    def __init__(
        self,
        name: str = "Mathematical Reasoning",
        difficulty: float = 0.7,
        llm_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize mathematical reasoning task.
        
        Args:
            name: Task name
            difficulty: Task difficulty level
            llm_config: Configuration for LLM interface
        """
        super().__init__(
            name=name,
            description="Evaluate mathematical reasoning and problem-solving capabilities",
            domain="mathematical_reasoning",
            difficulty=difficulty
        )
        
        self.llm = LLMInterface(llm_config or {'provider': 'mock'})
        
        # Test case categories
        self.test_categories = [
            'arithmetic',
            'algebra',
            'word_problems',
            'multi_step',
            'geometry'
        ]
        
        # Generate test cases
        self.test_cases = self._generate_test_cases()
    
    def evaluate(self, prompt: str, **kwargs) -> float:
        """
        Evaluate prompt on mathematical reasoning tasks.
        
        Args:
            prompt: Prompt to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Score between 0.0 and 1.0
        """
        # Validate prompt format
        is_valid, error_msg = self.validate_prompt_format(prompt)
        if not is_valid:
            self.logger.warning(f"Invalid prompt format: {error_msg}")
            return 0.0
        
        # Select test cases for evaluation
        num_test_cases = kwargs.get('num_test_cases', 10)
        selected_cases = random.sample(self.test_cases, min(num_test_cases, len(self.test_cases)))
        
        scores = []
        
        for test_case in selected_cases:
            case_score = self._evaluate_test_case(prompt, test_case)
            scores.append(case_score)
        
        # Calculate overall score
        overall_score = np.mean(scores) if scores else 0.0
        
        # Apply difficulty adjustment
        return self.get_difficulty_adjusted_score(overall_score)
    
    def _evaluate_test_case(self, prompt: str, test_case: Dict[str, Any]) -> float:
        """Evaluate prompt on a single test case."""
        try:
            # Create full prompt with test case
            full_prompt = f"{prompt}\n\nProblem: {test_case['problem']}"
            
            # Get LLM response
            response = self.llm.generate([
                {"role": "user", "content": full_prompt}
            ])
            
            # Extract answer from response
            extracted_answer = self._extract_answer(response)
            expected_answer = test_case['answer']
            
            # Calculate score based on answer correctness
            score = self._calculate_answer_score(extracted_answer, expected_answer, test_case)
            
            # Bonus for showing work/reasoning
            reasoning_bonus = self._evaluate_reasoning_quality(response, test_case)
            
            # Combine scores
            final_score = min(1.0, score + reasoning_bonus * 0.2)
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error evaluating test case: {e}")
            return 0.0
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract the mathematical answer from LLM response."""
        # Look for common answer patterns
        patterns = [
            r'(?:answer|solution|result)(?:\s*is)?:?\s*([+-]?\d*\.?\d+)',
            r'(?:equals?|=)\s*([+-]?\d*\.?\d+)',
            r'([+-]?\d*\.?\d+)(?:\s*$|\s*\.$)',
            r'(?:^|\s)([+-]?\d*\.?\d+)(?=\s|$)'
        ]
        
        response_lower = response.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, response_lower)
            if matches:
                # Return the last match (likely the final answer)
                return matches[-1]
        
        return None
    
    def _calculate_answer_score(
        self, 
        extracted_answer: Optional[str], 
        expected_answer: str, 
        test_case: Dict[str, Any]
    ) -> float:
        """Calculate score based on answer correctness."""
        if extracted_answer is None:
            return 0.0
        
        try:
            extracted_num = float(extracted_answer)
            expected_num = float(expected_answer)
            
            # Check for exact match
            if abs(extracted_num - expected_num) < 1e-6:
                return 1.0
            
            # Check for close match (within tolerance)
            tolerance = test_case.get('tolerance', 0.01)
            relative_error = abs(extracted_num - expected_num) / max(abs(expected_num), 1e-6)
            
            if relative_error <= tolerance:
                return 0.8  # Partial credit for close answers
            
            # Check if at least in right ballpark
            if relative_error <= 0.1:
                return 0.3
            
            return 0.0
            
        except ValueError:
            # Non-numeric answers
            if str(extracted_answer).strip().lower() == str(expected_answer).strip().lower():
                return 1.0
            return 0.0
    
    def _evaluate_reasoning_quality(self, response: str, test_case: Dict[str, Any]) -> float:
        """Evaluate the quality of mathematical reasoning shown."""
        reasoning_indicators = [
            'step', 'first', 'then', 'next', 'therefore', 'because',
            'since', 'given', 'solve', 'substitute', 'calculate',
            'multiply', 'divide', 'add', 'subtract'
        ]
        
        response_lower = response.lower()
        
        # Count reasoning indicators
        indicator_count = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
        
        # Normalize score (0 to 1)
        reasoning_score = min(1.0, indicator_count / 5.0)
        
        # Check for mathematical symbols/operations
        math_symbols = ['+', '-', '*', '/', '=', '(', ')', '^']
        symbol_count = sum(1 for symbol in math_symbols if symbol in response)
        symbol_score = min(1.0, symbol_count / 10.0)
        
        # Combine scores
        return (reasoning_score + symbol_score) / 2.0
    
    def _generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate test cases for mathematical reasoning."""
        test_cases = []
        
        # Arithmetic problems
        test_cases.extend(self._generate_arithmetic_cases())
        
        # Algebraic problems
        test_cases.extend(self._generate_algebra_cases())
        
        # Word problems
        test_cases.extend(self._generate_word_problems())
        
        # Multi-step problems
        test_cases.extend(self._generate_multi_step_cases())
        
        # Geometry problems
        test_cases.extend(self._generate_geometry_cases())
        
        return test_cases
    
    def _generate_arithmetic_cases(self) -> List[Dict[str, Any]]:
        """Generate arithmetic test cases."""
        cases = []
        
        # Basic operations
        for _ in range(5):
            a, b = random.randint(1, 100), random.randint(1, 100)
            cases.append({
                'category': 'arithmetic',
                'problem': f"Calculate {a} + {b}",
                'answer': str(a + b),
                'tolerance': 0.0
            })
        
        # Multiplication and division
        for _ in range(5):
            a, b = random.randint(2, 20), random.randint(2, 20)
            cases.append({
                'category': 'arithmetic',
                'problem': f"What is {a} × {b}?",
                'answer': str(a * b),
                'tolerance': 0.0
            })
        
        # Fractions
        cases.append({
            'category': 'arithmetic',
            'problem': "Calculate 3/4 + 1/6",
            'answer': str(3/4 + 1/6),
            'tolerance': 0.001
        })
        
        return cases
    
    def _generate_algebra_cases(self) -> List[Dict[str, Any]]:
        """Generate algebraic test cases."""
        cases = []
        
        # Linear equations
        cases.append({
            'category': 'algebra',
            'problem': "Solve for x: 2x + 5 = 13",
            'answer': "4",
            'tolerance': 0.001
        })
        
        cases.append({
            'category': 'algebra',
            'problem': "If 3x - 7 = 14, what is x?",
            'answer': "7",
            'tolerance': 0.001
        })
        
        # Quadratic equations
        cases.append({
            'category': 'algebra',
            'problem': "Solve x² - 5x + 6 = 0 (give the smaller solution)",
            'answer': "2",
            'tolerance': 0.001
        })
        
        return cases
    
    def _generate_word_problems(self) -> List[Dict[str, Any]]:
        """Generate word problem test cases."""
        cases = []
        
        cases.append({
            'category': 'word_problems',
            'problem': "Sarah has 24 apples. She gives away 1/3 of them to her friends. How many apples does she have left?",
            'answer': "16",
            'tolerance': 0.0
        })
        
        cases.append({
            'category': 'word_problems',
            'problem': "A train travels 180 miles in 3 hours. What is its average speed in miles per hour?",
            'answer': "60",
            'tolerance': 0.0
        })
        
        cases.append({
            'category': 'word_problems',
            'problem': "John bought 3 books for $12 each and 2 pens for $3 each. What was the total cost?",
            'answer': "42",
            'tolerance': 0.0
        })
        
        return cases
    
    def _generate_multi_step_cases(self) -> List[Dict[str, Any]]:
        """Generate multi-step reasoning test cases."""
        cases = []
        
        cases.append({
            'category': 'multi_step',
            'problem': "A rectangle has length 8 and width 5. What is its perimeter? Then, if you increase both dimensions by 2, what is the new area?",
            'answer': "70",  # New area: 10 × 7 = 70
            'tolerance': 0.0
        })
        
        cases.append({
            'category': 'multi_step',
            'problem': "Start with 100. Subtract 25, then multiply by 3, then divide by 5. What is the result?",
            'answer': "45",
            'tolerance': 0.0
        })
        
        return cases
    
    def _generate_geometry_cases(self) -> List[Dict[str, Any]]:
        """Generate geometry test cases."""
        cases = []
        
        cases.append({
            'category': 'geometry',
            'problem': "What is the area of a circle with radius 5? (Use π ≈ 3.14159)",
            'answer': str(3.14159 * 25),
            'tolerance': 0.1
        })
        
        cases.append({
            'category': 'geometry',
            'problem': "A right triangle has legs of length 3 and 4. What is the length of the hypotenuse?",
            'answer': "5",
            'tolerance': 0.001
        })
        
        return cases
    
    def get_test_cases(self) -> List[Dict[str, Any]]:
        """Get all test cases for this task."""
        return self.test_cases.copy()
    
    def get_examples(self, n: int = 3) -> List[str]:
        """Get example problems and solutions."""
        examples = []
        
        sample_cases = random.sample(self.test_cases, min(n, len(self.test_cases)))
        
        for case in sample_cases:
            example = f"Problem: {case['problem']}\nAnswer: {case['answer']}"
            examples.append(example)
        
        return examples
    
    def validate_prompt_format(self, prompt: str) -> tuple:
        """Validate prompt format for mathematical reasoning."""
        is_valid, error_msg = super().validate_prompt_format(prompt)
        
        if not is_valid:
            return is_valid, error_msg
        
        # Check for mathematical reasoning keywords
        math_keywords = [
            'solve', 'calculate', 'find', 'determine', 'compute',
            'step', 'show', 'work', 'reasoning', 'explain'
        ]
        
        prompt_lower = prompt.lower()
        has_math_keywords = any(keyword in prompt_lower for keyword in math_keywords)
        
        if not has_math_keywords:
            return False, "Prompt should include mathematical reasoning instructions"
        
        return True, None
