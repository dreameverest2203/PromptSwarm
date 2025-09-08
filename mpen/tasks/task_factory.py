"""
Task factory for creating and managing MPEN evaluation tasks.
"""

from typing import Dict, Any, List, Optional, Type
import logging

from .base import Task
from .math_reasoning import MathReasoningTask
from .creative_writing import CreativeWritingTask
from .programming import ProgrammingTask


class TaskFactory:
    """
    Factory class for creating and managing evaluation tasks.
    
    Provides centralized task creation, configuration, and management
    for the MPEN system.
    """
    
    def __init__(self):
        """Initialize task factory."""
        self.logger = logging.getLogger("mpen.task_factory")
        
        # Registry of available task types
        self.task_registry: Dict[str, Type[Task]] = {
            'math_reasoning': MathReasoningTask,
            'mathematical_reasoning': MathReasoningTask,
            'creative_writing': CreativeWritingTask,
            'programming': ProgrammingTask,
            'coding': ProgrammingTask
        }
        
        # Default configurations for each task type
        self.default_configs: Dict[str, Dict[str, Any]] = {
            'math_reasoning': {
                'difficulty': 0.7,
                'name': 'Mathematical Reasoning'
            },
            'creative_writing': {
                'difficulty': 0.6,
                'name': 'Creative Writing'
            },
            'programming': {
                'difficulty': 0.8,
                'name': 'Programming'
            }
        }
        
        self.logger.info("Initialized task factory with task types: %s", list(self.task_registry.keys()))
    
    def create_task(
        self,
        task_type: str,
        config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None
    ) -> Task:
        """
        Create a task instance.
        
        Args:
            task_type: Type of task to create
            config: Task-specific configuration
            llm_config: LLM configuration for the task
            
        Returns:
            Task instance
            
        Raises:
            ValueError: If task type is not supported
        """
        task_type = task_type.lower()
        
        if task_type not in self.task_registry:
            available_types = list(self.task_registry.keys())
            raise ValueError(f"Unknown task type: {task_type}. Available types: {available_types}")
        
        # Get task class
        task_class = self.task_registry[task_type]
        
        # Merge configurations
        final_config = self.default_configs.get(task_type, {}).copy()
        if config:
            final_config.update(config)
        
        # Add LLM config if provided
        if llm_config:
            final_config['llm_config'] = llm_config
        
        # Create task instance
        task_instance = task_class(**final_config)
        
        self.logger.info(f"Created task: {task_instance.name} (type: {task_type})")
        
        return task_instance
    
    def create_task_suite(
        self,
        task_configs: List[Dict[str, Any]],
        global_llm_config: Optional[Dict[str, Any]] = None
    ) -> List[Task]:
        """
        Create a suite of tasks.
        
        Args:
            task_configs: List of task configuration dictionaries
            global_llm_config: Global LLM configuration
            
        Returns:
            List of task instances
        """
        tasks = []
        
        for task_config in task_configs:
            if 'type' not in task_config:
                self.logger.warning("Skipping task config without 'type' field")
                continue
            
            task_type = task_config.pop('type')
            
            # Use global LLM config if not specified in task config
            llm_config = task_config.pop('llm_config', global_llm_config)
            
            try:
                task = self.create_task(task_type, task_config, llm_config)
                tasks.append(task)
            except Exception as e:
                self.logger.error(f"Failed to create task of type {task_type}: {e}")
        
        self.logger.info(f"Created task suite with {len(tasks)} tasks")
        return tasks
    
    def get_available_task_types(self) -> List[str]:
        """Get list of available task types."""
        return list(self.task_registry.keys())
    
    def register_task_type(
        self,
        task_type: str,
        task_class: Type[Task],
        default_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new task type.
        
        Args:
            task_type: Name for the task type
            task_class: Task class to register
            default_config: Default configuration for the task type
        """
        self.task_registry[task_type.lower()] = task_class
        
        if default_config:
            self.default_configs[task_type.lower()] = default_config
        
        self.logger.info(f"Registered new task type: {task_type}")
    
    def create_benchmark_suite(
        self,
        difficulty_level: str = 'medium',
        llm_config: Optional[Dict[str, Any]] = None
    ) -> List[Task]:
        """
        Create a standard benchmark suite.
        
        Args:
            difficulty_level: 'easy', 'medium', or 'hard'
            llm_config: LLM configuration
            
        Returns:
            List of benchmark tasks
        """
        # Define difficulty mappings
        difficulty_mappings = {
            'easy': 0.4,
            'medium': 0.6,
            'hard': 0.8
        }
        
        difficulty_value = difficulty_mappings.get(difficulty_level, 0.6)
        
        # Create standard benchmark tasks
        benchmark_configs = [
            {
                'type': 'math_reasoning',
                'name': f'Math Reasoning ({difficulty_level.title()})',
                'difficulty': difficulty_value
            },
            {
                'type': 'creative_writing',
                'name': f'Creative Writing ({difficulty_level.title()})',
                'difficulty': difficulty_value
            },
            {
                'type': 'programming',
                'name': f'Programming ({difficulty_level.title()})',
                'difficulty': difficulty_value
            }
        ]
        
        return self.create_task_suite(benchmark_configs, llm_config)
    
    def create_domain_specific_suite(
        self,
        domain: str,
        llm_config: Optional[Dict[str, Any]] = None
    ) -> List[Task]:
        """
        Create a domain-specific task suite.
        
        Args:
            domain: Domain name ('math', 'writing', 'programming')
            llm_config: LLM configuration
            
        Returns:
            List of domain-specific tasks
        """
        domain_mappings = {
            'math': ['math_reasoning'],
            'mathematics': ['math_reasoning'],
            'writing': ['creative_writing'],
            'creative': ['creative_writing'],
            'programming': ['programming'],
            'coding': ['programming'],
            'code': ['programming']
        }
        
        domain = domain.lower()
        
        if domain not in domain_mappings:
            available_domains = list(domain_mappings.keys())
            raise ValueError(f"Unknown domain: {domain}. Available domains: {available_domains}")
        
        task_types = domain_mappings[domain]
        
        # Create tasks with different difficulty levels
        task_configs = []
        for task_type in task_types:
            for difficulty_name, difficulty_value in [('Easy', 0.4), ('Medium', 0.6), ('Hard', 0.8)]:
                task_configs.append({
                    'type': task_type,
                    'name': f'{task_type.title()} - {difficulty_name}',
                    'difficulty': difficulty_value
                })
        
        return self.create_task_suite(task_configs, llm_config)
    
    def create_custom_task(
        self,
        task_type: str,
        name: str,
        description: str,
        difficulty: float = 0.5,
        custom_config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None
    ) -> Task:
        """
        Create a custom task with specific parameters.
        
        Args:
            task_type: Type of task to create
            name: Custom name for the task
            description: Custom description
            difficulty: Task difficulty (0.0 to 1.0)
            custom_config: Additional custom configuration
            llm_config: LLM configuration
            
        Returns:
            Custom task instance
        """
        config = {
            'name': name,
            'description': description,
            'difficulty': difficulty
        }
        
        if custom_config:
            config.update(custom_config)
        
        return self.create_task(task_type, config, llm_config)
    
    def get_task_info(self, task_type: str) -> Dict[str, Any]:
        """
        Get information about a task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            Task type information
        """
        task_type = task_type.lower()
        
        if task_type not in self.task_registry:
            raise ValueError(f"Unknown task type: {task_type}")
        
        task_class = self.task_registry[task_type]
        default_config = self.default_configs.get(task_type, {})
        
        return {
            'task_type': task_type,
            'task_class': task_class.__name__,
            'description': task_class.__doc__ or "No description available",
            'default_config': default_config,
            'module': task_class.__module__
        }
    
    def list_task_types(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available task types with their information.
        
        Returns:
            Dictionary of task type information
        """
        task_info = {}
        
        for task_type in self.task_registry:
            try:
                task_info[task_type] = self.get_task_info(task_type)
            except Exception as e:
                self.logger.error(f"Error getting info for task type {task_type}: {e}")
        
        return task_info
    
    def validate_task_config(
        self,
        task_type: str,
        config: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a task configuration.
        
        Args:
            task_type: Type of task
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        task_type = task_type.lower()
        
        if task_type not in self.task_registry:
            return False, f"Unknown task type: {task_type}"
        
        # Check required fields
        required_fields = ['name']
        for field in required_fields:
            if field not in config:
                return False, f"Missing required field: {field}"
        
        # Validate difficulty range
        if 'difficulty' in config:
            difficulty = config['difficulty']
            if not isinstance(difficulty, (int, float)) or not (0.0 <= difficulty <= 1.0):
                return False, "Difficulty must be a number between 0.0 and 1.0"
        
        return True, None


# Global task factory instance
task_factory = TaskFactory()
