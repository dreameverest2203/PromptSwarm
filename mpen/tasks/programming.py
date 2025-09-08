"""
Programming task for MPEN evaluation.
"""

import re
import ast
import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .base import Task
from ..utils.llm_interface import LLMInterface


class ProgrammingTask(Task):
    """
    Programming task that evaluates prompts on code generation and programming problems.
    
    Tests various aspects of programming including:
    - Code correctness
    - Algorithm efficiency
    - Code structure and readability
    - Problem decomposition
    - Error handling
    - Documentation quality
    """
    
    def __init__(
        self,
        name: str = "Programming",
        difficulty: float = 0.8,
        llm_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize programming task.
        
        Args:
            name: Task name
            difficulty: Task difficulty level
            llm_config: Configuration for LLM interface
        """
        super().__init__(
            name=name,
            description="Evaluate programming and algorithm design capabilities",
            domain="programming",
            difficulty=difficulty
        )
        
        self.llm = LLMInterface(llm_config or {'provider': 'mock'})
        
        # Programming evaluation criteria
        self.evaluation_criteria = [
            'correctness',
            'efficiency',
            'readability',
            'structure',
            'error_handling',
            'documentation'
        ]
        
        # Generate test cases
        self.test_cases = self._generate_test_cases()
    
    def evaluate(self, prompt: str, **kwargs) -> float:
        """
        Evaluate prompt on programming tasks.
        
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
        num_test_cases = kwargs.get('num_test_cases', 5)
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
            
            if 'constraints' in test_case:
                full_prompt += f"\n\nConstraints: {test_case['constraints']}"
            
            # Get LLM response
            response = self.llm.generate([
                {"role": "user", "content": full_prompt}
            ])
            
            # Extract code from response
            extracted_code = self._extract_code(response)
            
            if not extracted_code:
                return 0.1  # Small score for attempting
            
            # Evaluate different aspects
            scores = {}
            
            for criterion in self.evaluation_criteria:
                criterion_score = self._evaluate_criterion(
                    extracted_code, response, test_case, criterion
                )
                scores[criterion] = criterion_score
            
            # Weight different criteria
            weights = {
                'correctness': 0.35,
                'efficiency': 0.20,
                'readability': 0.15,
                'structure': 0.15,
                'error_handling': 0.10,
                'documentation': 0.05
            }
            
            # Calculate weighted score
            weighted_score = sum(
                scores[criterion] * weights.get(criterion, 1.0)
                for criterion in scores
            )
            
            return min(1.0, weighted_score)
            
        except Exception as e:
            self.logger.error(f"Error evaluating test case: {e}")
            return 0.0
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract code from LLM response."""
        # Look for code blocks
        code_patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'`([^`]+)`',
            r'def\s+\w+\([^)]*\):[^}]+',
            r'class\s+\w+[^:]*:[^}]+'
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                # Return the longest match (likely the main code)
                return max(matches, key=len)
        
        # If no code blocks found, look for function definitions
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if any(keyword in line for keyword in ['def ', 'class ', 'import ', 'from ']):
                in_code = True
            
            if in_code:
                code_lines.append(line)
                
                # Stop at empty line or non-code content
                if not line.strip() and code_lines:
                    break
        
        return '\n'.join(code_lines) if code_lines else None
    
    def _evaluate_criterion(
        self,
        code: str,
        full_response: str,
        test_case: Dict[str, Any],
        criterion: str
    ) -> float:
        """Evaluate a specific criterion."""
        if criterion == 'correctness':
            return self._evaluate_correctness(code, test_case)
        elif criterion == 'efficiency':
            return self._evaluate_efficiency(code, test_case)
        elif criterion == 'readability':
            return self._evaluate_readability(code)
        elif criterion == 'structure':
            return self._evaluate_structure(code)
        elif criterion == 'error_handling':
            return self._evaluate_error_handling(code)
        elif criterion == 'documentation':
            return self._evaluate_documentation(code, full_response)
        else:
            return 0.5
    
    def _evaluate_correctness(self, code: str, test_case: Dict[str, Any]) -> float:
        """Evaluate code correctness."""
        try:
            # Basic syntax check
            ast.parse(code)
            syntax_score = 1.0
        except SyntaxError:
            return 0.0
        
        # Check for required function/class names
        required_names = test_case.get('required_names', [])
        name_score = 0.0
        
        if required_names:
            found_names = sum(1 for name in required_names if name in code)
            name_score = found_names / len(required_names)
        else:
            name_score = 0.5  # Default if no specific names required
        
        # Check for algorithm patterns
        algorithm_type = test_case.get('algorithm_type')
        algorithm_score = self._check_algorithm_pattern(code, algorithm_type)
        
        # Check for expected operations
        expected_operations = test_case.get('expected_operations', [])
        operation_score = 0.0
        
        if expected_operations:
            found_operations = sum(1 for op in expected_operations if op in code)
            operation_score = found_operations / len(expected_operations)
        else:
            operation_score = 0.5
        
        # Combine correctness factors
        correctness_score = (syntax_score + name_score + algorithm_score + operation_score) / 4.0
        
        return correctness_score
    
    def _check_algorithm_pattern(self, code: str, algorithm_type: Optional[str]) -> float:
        """Check for specific algorithm patterns."""
        if not algorithm_type:
            return 0.5
        
        code_lower = code.lower()
        
        if algorithm_type == 'sorting':
            sorting_patterns = ['sort', 'sorted', 'bubble', 'quick', 'merge']
            return 1.0 if any(pattern in code_lower for pattern in sorting_patterns) else 0.3
        
        elif algorithm_type == 'searching':
            search_patterns = ['search', 'find', 'binary', 'linear', 'in ']
            return 1.0 if any(pattern in code_lower for pattern in search_patterns) else 0.3
        
        elif algorithm_type == 'recursion':
            return 1.0 if 'def ' in code and code.count('def ') > 0 and any(
                func_name in code for func_name in re.findall(r'def\s+(\w+)', code)
            ) else 0.3
        
        elif algorithm_type == 'iteration':
            iteration_patterns = ['for ', 'while ', 'range(']
            return 1.0 if any(pattern in code_lower for pattern in iteration_patterns) else 0.3
        
        return 0.5
    
    def _evaluate_efficiency(self, code: str, test_case: Dict[str, Any]) -> float:
        """Evaluate algorithm efficiency."""
        # Simple heuristics for efficiency
        efficiency_score = 1.0
        
        # Penalize nested loops (potential O(n²) or worse)
        nested_loop_penalty = 0.0
        for_count = code.count('for ')
        while_count = code.count('while ')
        
        if for_count > 1 or while_count > 1:
            nested_loop_penalty = 0.2
        
        # Reward efficient operations
        efficient_operations = ['sort()', 'sorted()', 'set()', 'dict()', 'enumerate()']
        efficiency_bonus = sum(0.1 for op in efficient_operations if op in code)
        efficiency_bonus = min(0.3, efficiency_bonus)
        
        # Penalize inefficient patterns
        inefficient_patterns = ['list.append()', '.insert(0,', 'del list[0]']
        inefficiency_penalty = sum(0.1 for pattern in inefficient_patterns if pattern in code)
        inefficiency_penalty = min(0.3, inefficiency_penalty)
        
        # Calculate final efficiency score
        efficiency_score = efficiency_score - nested_loop_penalty + efficiency_bonus - inefficiency_penalty
        
        return max(0.0, min(1.0, efficiency_score))
    
    def _evaluate_readability(self, code: str) -> float:
        """Evaluate code readability."""
        lines = [line for line in code.split('\n') if line.strip()]
        
        if not lines:
            return 0.0
        
        # Check for meaningful variable names
        variable_names = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=', code)
        meaningful_names = sum(1 for name in variable_names if len(name) > 2 and name not in ['i', 'j', 'k'])
        name_score = meaningful_names / len(variable_names) if variable_names else 0.5
        
        # Check for appropriate spacing
        spacing_score = 1.0
        if '=' in code:
            # Look for spaces around operators
            good_spacing = code.count(' = ') + code.count(' == ') + code.count(' != ')
            total_operators = code.count('=')
            spacing_score = good_spacing / total_operators if total_operators > 0 else 1.0
        
        # Check for reasonable line length
        long_lines = sum(1 for line in lines if len(line) > 100)
        line_length_score = max(0.0, 1.0 - (long_lines / len(lines)))
        
        # Check for comments
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        comment_score = min(1.0, comment_lines / max(1, len(lines) // 5))
        
        # Combine readability factors
        readability_score = (name_score + spacing_score + line_length_score + comment_score) / 4.0
        
        return readability_score
    
    def _evaluate_structure(self, code: str) -> float:
        """Evaluate code structure and organization."""
        # Check for function definitions
        function_count = len(re.findall(r'def\s+\w+', code))
        function_score = min(1.0, function_count / 2.0)  # Reward up to 2 functions
        
        # Check for class definitions
        class_count = len(re.findall(r'class\s+\w+', code))
        class_score = min(0.3, class_count * 0.3)
        
        # Check for proper indentation
        lines = code.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
        indentation_score = indented_lines / len(lines) if lines else 0
        
        # Check for imports at top
        import_lines = [i for i, line in enumerate(lines) if line.strip().startswith(('import ', 'from '))]
        import_score = 1.0 if not import_lines or all(i < 5 for i in import_lines) else 0.5
        
        # Combine structure factors
        structure_score = (function_score + class_score + indentation_score + import_score) / 4.0
        
        return min(1.0, structure_score)
    
    def _evaluate_error_handling(self, code: str) -> float:
        """Evaluate error handling."""
        error_handling_patterns = ['try:', 'except:', 'raise', 'assert', 'if not', 'is None']
        
        code_lower = code.lower()
        error_handling_count = sum(1 for pattern in error_handling_patterns if pattern in code_lower)
        
        # Score based on presence of error handling
        if error_handling_count == 0:
            return 0.3  # Basic score for no explicit error handling
        elif error_handling_count <= 2:
            return 0.7  # Good score for some error handling
        else:
            return 1.0  # Excellent score for comprehensive error handling
    
    def _evaluate_documentation(self, code: str, full_response: str) -> float:
        """Evaluate code documentation."""
        # Check for docstrings
        docstring_count = len(re.findall(r'""".*?"""', code, re.DOTALL))
        docstring_count += len(re.findall(r"'''.*?'''", code, re.DOTALL))
        
        # Check for inline comments
        comment_lines = sum(1 for line in code.split('\n') if '#' in line)
        
        # Check for explanation in response
        explanation_keywords = ['explanation', 'this function', 'algorithm', 'approach', 'solution']
        explanation_count = sum(1 for keyword in explanation_keywords if keyword in full_response.lower())
        
        # Combine documentation factors
        docstring_score = min(1.0, docstring_count * 0.5)
        comment_score = min(1.0, comment_lines / max(1, len(code.split('\n')) // 3))
        explanation_score = min(1.0, explanation_count / 2.0)
        
        documentation_score = (docstring_score + comment_score + explanation_score) / 3.0
        
        return documentation_score
    
    def _generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate test cases for programming evaluation."""
        test_cases = []
        
        # Basic algorithm problems
        test_cases.extend([
            {
                'problem': 'Write a function to find the maximum element in a list.',
                'algorithm_type': 'searching',
                'required_names': ['max', 'maximum'],
                'expected_operations': ['for ', 'max('],
                'difficulty': 0.3
            },
            {
                'problem': 'Implement a function to reverse a string.',
                'algorithm_type': 'iteration',
                'required_names': ['reverse'],
                'expected_operations': ['[::-1]', 'reversed('],
                'difficulty': 0.4
            },
            {
                'problem': 'Write a function to check if a number is prime.',
                'algorithm_type': 'iteration',
                'required_names': ['prime', 'is_prime'],
                'expected_operations': ['for ', 'range(', '%'],
                'difficulty': 0.6
            }
        ])
        
        # Sorting problems
        test_cases.extend([
            {
                'problem': 'Implement bubble sort algorithm.',
                'algorithm_type': 'sorting',
                'required_names': ['bubble_sort', 'sort'],
                'expected_operations': ['for ', 'while ', 'swap'],
                'difficulty': 0.7
            },
            {
                'problem': 'Write a function to merge two sorted lists.',
                'algorithm_type': 'sorting',
                'required_names': ['merge'],
                'expected_operations': ['while ', 'append'],
                'difficulty': 0.6
            }
        ])
        
        # Data structure problems
        test_cases.extend([
            {
                'problem': 'Implement a stack using a list.',
                'algorithm_type': 'data_structure',
                'required_names': ['Stack', 'push', 'pop'],
                'expected_operations': ['class ', 'def ', 'append', 'pop'],
                'difficulty': 0.7
            },
            {
                'problem': 'Write a function to find duplicate elements in a list.',
                'algorithm_type': 'searching',
                'required_names': ['duplicate', 'find'],
                'expected_operations': ['set(', 'dict(', 'count'],
                'difficulty': 0.5
            }
        ])
        
        # Recursive problems
        test_cases.extend([
            {
                'problem': 'Calculate factorial using recursion.',
                'algorithm_type': 'recursion',
                'required_names': ['factorial'],
                'expected_operations': ['def ', 'return', '*'],
                'difficulty': 0.6
            },
            {
                'problem': 'Implement Fibonacci sequence using recursion.',
                'algorithm_type': 'recursion',
                'required_names': ['fibonacci', 'fib'],
                'expected_operations': ['def ', 'return', '+'],
                'difficulty': 0.7
            }
        ])
        
        # String manipulation
        test_cases.extend([
            {
                'problem': 'Write a function to count vowels in a string.',
                'algorithm_type': 'iteration',
                'required_names': ['vowel', 'count'],
                'expected_operations': ['for ', 'in ', 'lower()'],
                'difficulty': 0.4
            },
            {
                'problem': 'Check if a string is a palindrome.',
                'algorithm_type': 'iteration',
                'required_names': ['palindrome'],
                'expected_operations': ['lower()', '==', '[::-1]'],
                'difficulty': 0.5
            }
        ])
        
        return test_cases
    
    def get_test_cases(self) -> List[Dict[str, Any]]:
        """Get all test cases for this task."""
        return self.test_cases.copy()
    
    def validate_prompt_format(self, prompt: str) -> Tuple[bool, Optional[str]]:
        """Validate prompt format for programming tasks."""
        is_valid, error_msg = super().validate_prompt_format(prompt)
        
        if not is_valid:
            return is_valid, error_msg
        
        # Check for programming keywords
        programming_keywords = [
            'function', 'implement', 'write', 'code', 'algorithm',
            'program', 'solve', 'create', 'design', 'develop'
        ]
        
        prompt_lower = prompt.lower()
        has_programming_keywords = any(keyword in prompt_lower for keyword in programming_keywords)
        
        if not has_programming_keywords:
            return False, "Prompt should include programming-related instructions"
        
        return True, None
