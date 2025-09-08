# Meta-Prompt Evolutionary Networks (MPEN)

A novel approach to prompt optimization using collaborative multi-agent systems with adaptive network connections.

## Abstract

Current automated prompt optimization methods hit a wall when dealing with complex, multi-objective problems. Most approaches rely on a single model iteratively refining prompts, but this creates bottlenecks and struggles with the competing demands of accuracy, efficiency, and safety that real applications require.

We propose Meta-Prompt Evolutionary Networks (MPEN), where multiple specialized language model agents work together to evolve better prompts. Rather than one model doing everything, we deploy four types of agents: generators that create prompt variations, critics that evaluate different aspects like bias and clarity, validators that test on unseen tasks, and meta-agents that coordinate the whole process.

## Key Features

- **Multi-Agent Architecture**: Four specialized agent types working collaboratively
- **Adaptive Networks**: Agent connections that strengthen/weaken based on collaboration success
- **Domain Transfer**: Learned patterns transfer across different task domains
- **Game Theory Coordination**: Theoretical foundation for cooperation vs competition
- **Interpretable Process**: Explicit agent roles make optimization transparent

## Agent Types

1. **Generator Agents**: Create diverse prompt variations and mutations
2. **Critic Agents**: Evaluate prompts across multiple dimensions (bias, clarity, safety)
3. **Validator Agents**: Test prompts on unseen tasks and edge cases
4. **Meta Agents**: Coordinate the network and manage global optimization

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from mpen import MPENSystem
from mpen.tasks import MathReasoningTask

# Initialize the MPEN system
system = MPENSystem(
    num_generators=3,
    num_critics=2,
    num_validators=2,
    num_meta_agents=1
)

# Define a task
task = MathReasoningTask("Solve multi-step algebra problems")

# Run optimization
optimized_prompt = system.optimize(
    initial_prompt="Solve the following problem step by step:",
    task=task,
    max_iterations=50
)

print(f"Optimized prompt: {optimized_prompt}")
```

## Evaluation Domains

- **Mathematical Reasoning**: Multi-step algebra, geometry, calculus
- **Creative Writing**: Story generation, poetry, dialogue
- **Multi-step Programming**: Algorithm design, code optimization, debugging

## Project Structure

```
mpen/
├── agents/              # Agent implementations
├── network/             # Network connection management
├── evolutionary/        # Evolutionary optimization framework
├── tasks/              # Evaluation tasks and benchmarks
├── coordination/       # Game theory coordination
├── experiments/        # Experimental configurations
└── utils/              # Utilities and helpers
```

## Research Paper

This implementation supports the research presented in our workshop paper on collaborative prompt optimization. The system demonstrates how specialized agents can achieve better results than single-model approaches while maintaining interpretability.

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@inproceedings{mpen2024,
  title={Meta-Prompt Evolutionary Networks: Collaborative Multi-Agent Prompt Optimization},
  author={[Your Name]},
  booktitle={Workshop on [Conference Name]},
  year={2024}
}
```
