# Contributing to MPEN

We welcome contributions to the Meta-Prompt Evolutionary Networks (MPEN) project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/promptswarm.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Install in development mode: `pip install -e .`

## Development Setup

### Code Style

We use the following tools for code quality:

- **Black** for code formatting: `black mpen/`
- **Flake8** for linting: `flake8 mpen/`
- **MyPy** for type checking: `mypy mpen/`

Run these before submitting pull requests.

### Testing

Run tests with pytest:
```bash
pytest tests/
```

Add tests for new functionality in the `tests/` directory.

## Contributing Guidelines

### Types of Contributions

1. **Bug Reports**: Use the issue tracker with the "bug" label
2. **Feature Requests**: Use the issue tracker with the "enhancement" label  
3. **Code Contributions**: Submit pull requests
4. **Documentation**: Improve docs, examples, or README
5. **Research**: New agent types, coordination strategies, or evaluation tasks

### Code Contributions

1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Make changes**: Follow the existing code style and patterns
3. **Add tests**: Ensure your changes are tested
4. **Update docs**: Update docstrings and README if needed
5. **Commit**: Use clear, descriptive commit messages
6. **Push**: `git push origin feature/your-feature-name`
7. **Pull Request**: Create a PR with a clear description

### Commit Messages

Use clear, descriptive commit messages:
- `feat: add new generator agent mutation strategy`
- `fix: resolve network connection pruning bug`
- `docs: update API documentation for tasks module`
- `test: add tests for evolutionary optimizer`

### Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add yourself to CONTRIBUTORS.md
4. Fill out the pull request template
5. Request review from maintainers

## Code Organization

The project is organized as follows:

```
mpen/
├── agents/          # Agent implementations
├── network/         # Network connection system
├── evolutionary/    # Evolutionary optimization
├── tasks/          # Evaluation tasks
├── coordination/   # Game theory coordination
├── utils/          # Utilities and helpers
examples/           # Usage examples
experiments/        # Experiment configurations
tests/             # Test suite
```

### Adding New Components

#### New Agent Types

1. Inherit from `BaseAgent` in `mpen/agents/base.py`
2. Implement required methods: `process()`, `_collaborate_impl()`
3. Add specialization domains and metrics tracking
4. Add tests in `tests/agents/`
5. Update `mpen/agents/__init__.py`

#### New Tasks

1. Inherit from `Task` in `mpen/tasks/base.py`
2. Implement `evaluate()` and `get_test_cases()`
3. Add domain-specific evaluation logic
4. Add tests in `tests/tasks/`
5. Register in `TaskFactory`

#### New Coordination Strategies

1. Add to `mpen/coordination/`
2. Implement game theory principles
3. Provide convergence guarantees where applicable
4. Add comprehensive tests

### Documentation

- Use Google-style docstrings
- Include type hints
- Provide examples in docstrings
- Update README for major changes
- Add examples for new features

## Research Contributions

We especially welcome research contributions:

### New Agent Types
- Specialized agents for specific domains
- Novel collaboration patterns
- Improved reasoning capabilities

### Coordination Mechanisms
- Advanced game theory applications
- Multi-objective optimization
- Distributed consensus algorithms

### Evaluation Tasks
- New domains (science, law, medicine, etc.)
- Challenging benchmarks
- Real-world applications

### Theoretical Analysis
- Convergence proofs
- Complexity analysis
- Comparative studies

## Experimental Guidelines

When adding experiments:

1. Use the experiment framework in `experiments/`
2. Follow reproducibility standards
3. Include statistical significance testing
4. Document experimental setup clearly
5. Provide visualization code

## Community

- Be respectful and inclusive
- Help others learn and contribute
- Share interesting results and findings
- Participate in discussions and reviews

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for research questions
- Contact maintainers for collaboration opportunities

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Research paper acknowledgments
- Release notes for significant contributions

Thank you for contributing to MPEN!
