#!/usr/bin/env python3
"""
Basic usage example for MPEN system.

This script demonstrates how to:
1. Initialize the MPEN system
2. Create tasks for evaluation
3. Run prompt optimization
4. Analyze results
"""

import sys
import os

# Add the parent directory to path to import mpen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mpen import MPENSystem
from mpen.tasks import TaskFactory
from mpen.utils.logging import setup_logger


def main():
    """Run basic MPEN usage example."""
    
    # Set up logging
    logger = setup_logger("mpen.example", level="INFO")
    logger.info("Starting MPEN basic usage example")
    
    # Configure LLM (using mock for demonstration)
    llm_config = {
        'provider': 'mock',
        'mock_responses': [
            "I'll solve this step by step. First, I need to understand the problem...",
            "Let me break this down systematically and provide a clear solution...",
            "Here's my approach to solving this problem with detailed reasoning..."
        ]
    }
    
    # Initialize MPEN system
    logger.info("Initializing MPEN system...")
    system = MPENSystem(
        num_generators=2,
        num_critics=2,
        num_validators=1,
        num_meta_agents=1,
        llm_config=llm_config
    )
    
    # Create task factory and tasks
    logger.info("Creating evaluation tasks...")
    task_factory = TaskFactory()
    
    # Create a mathematical reasoning task
    math_task = task_factory.create_task(
        'math_reasoning',
        config={'difficulty': 0.6},
        llm_config=llm_config
    )
    
    # Define initial prompt to optimize
    initial_prompt = "Solve the following problem step by step:"
    
    logger.info("Starting prompt optimization...")
    logger.info(f"Initial prompt: '{initial_prompt}'")
    logger.info(f"Task: {math_task.name} (difficulty: {math_task.difficulty})")
    
    # Run optimization
    try:
        result = system.optimize(
            initial_prompt=initial_prompt,
            task=math_task,
            max_iterations=10,
            population_size=8
        )
        
        # Display results
        logger.info("Optimization completed!")
        logger.info(f"Best prompt: '{result.best_prompt}'")
        logger.info(f"Best score: {result.best_score:.4f}")
        logger.info(f"Total iterations: {len(result.iteration_history)}")
        
        # Show optimization progress
        logger.info("\nOptimization progress:")
        for i, iteration in enumerate(result.iteration_history[-5:], 1):  # Show last 5 iterations
            logger.info(f"  Iteration {iteration['iteration']}: "
                       f"best={iteration['best_score']:.4f}, "
                       f"mean={iteration['mean_score']:.4f}")
        
        # Show agent contributions
        logger.info("\nAgent contributions:")
        for agent_id, contribution in result.agent_contributions.items():
            logger.info(f"  {agent_id}: {contribution:.3f}")
        
        # Show network statistics
        network_viz = system.get_network_visualization()
        logger.info(f"\nNetwork statistics:")
        logger.info(f"  Total connections: {len(network_viz['edges'])}")
        logger.info(f"  Active agents: {len(network_viz['nodes'])}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1
    
    logger.info("Example completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
