#!/usr/bin/env python3
"""
Multi-domain experiment example for MPEN system.

This script demonstrates how to:
1. Run experiments across multiple domains
2. Compare performance across different tasks
3. Analyze agent specialization patterns
4. Visualize network evolution
"""

import sys
import os
import time
from typing import Dict, List, Any

# Add the parent directory to path to import mpen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mpen import MPENSystem
from mpen.tasks import TaskFactory
from mpen.utils.logging import setup_logger


class MultiDomainExperiment:
    """Multi-domain experiment runner."""
    
    def __init__(self, llm_config: Dict[str, Any]):
        """Initialize experiment."""
        self.logger = setup_logger("mpen.experiment", level="INFO")
        self.llm_config = llm_config
        self.task_factory = TaskFactory()
        self.results: Dict[str, Any] = {}
    
    def run_experiment(
        self,
        domains: List[str],
        initial_prompts: Dict[str, str],
        max_iterations: int = 15,
        population_size: int = 12
    ) -> Dict[str, Any]:
        """Run multi-domain experiment."""
        
        self.logger.info("Starting multi-domain experiment")
        self.logger.info(f"Domains: {domains}")
        
        # Initialize MPEN system
        system = MPENSystem(
            num_generators=3,
            num_critics=2,
            num_validators=2,
            num_meta_agents=1,
            llm_config=self.llm_config
        )
        
        domain_results = {}
        
        for domain in domains:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Running experiment for domain: {domain}")
            self.logger.info(f"{'='*50}")
            
            # Create task for domain
            task = self._create_domain_task(domain)
            initial_prompt = initial_prompts.get(domain, "Complete the following task:")
            
            # Run optimization
            start_time = time.time()
            
            try:
                result = system.optimize(
                    initial_prompt=initial_prompt,
                    task=task,
                    max_iterations=max_iterations,
                    population_size=population_size
                )
                
                experiment_time = time.time() - start_time
                
                # Store results
                domain_results[domain] = {
                    'task_name': task.name,
                    'initial_prompt': initial_prompt,
                    'best_prompt': result.best_prompt,
                    'best_score': result.best_score,
                    'iterations': len(result.iteration_history),
                    'experiment_time': experiment_time,
                    'agent_contributions': result.agent_contributions,
                    'iteration_history': result.iteration_history,
                    'network_evolution': result.network_evolution
                }
                
                self.logger.info(f"Domain {domain} completed:")
                self.logger.info(f"  Best score: {result.best_score:.4f}")
                self.logger.info(f"  Time: {experiment_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Domain {domain} failed: {e}")
                domain_results[domain] = {
                    'error': str(e),
                    'task_name': task.name,
                    'initial_prompt': initial_prompt
                }
        
        # Analyze cross-domain patterns
        self.logger.info(f"\n{'='*50}")
        self.logger.info("Analyzing cross-domain patterns...")
        self.logger.info(f"{'='*50}")
        
        analysis = self._analyze_cross_domain_results(domain_results, system)
        
        # Compile final results
        final_results = {
            'domain_results': domain_results,
            'cross_domain_analysis': analysis,
            'system_config': {
                'max_iterations': max_iterations,
                'population_size': population_size,
                'num_agents': len(system.all_agents)
            },
            'experiment_metadata': {
                'domains': domains,
                'total_time': sum(
                    r.get('experiment_time', 0) 
                    for r in domain_results.values()
                ),
                'timestamp': time.time()
            }
        }
        
        self.results = final_results
        return final_results
    
    def _create_domain_task(self, domain: str):
        """Create appropriate task for domain."""
        domain_mappings = {
            'math': 'math_reasoning',
            'mathematics': 'math_reasoning',
            'writing': 'creative_writing',
            'creative': 'creative_writing',
            'programming': 'programming',
            'coding': 'programming'
        }
        
        task_type = domain_mappings.get(domain.lower(), domain)
        
        return self.task_factory.create_task(
            task_type,
            config={'difficulty': 0.7},
            llm_config=self.llm_config
        )
    
    def _analyze_cross_domain_results(
        self,
        domain_results: Dict[str, Any],
        system: MPENSystem
    ) -> Dict[str, Any]:
        """Analyze patterns across domains."""
        
        analysis = {}
        
        # Performance comparison
        successful_domains = {
            domain: results for domain, results in domain_results.items()
            if 'best_score' in results
        }
        
        if successful_domains:
            scores = [r['best_score'] for r in successful_domains.values()]
            analysis['performance'] = {
                'best_domain': max(successful_domains.items(), key=lambda x: x[1]['best_score'])[0],
                'worst_domain': min(successful_domains.items(), key=lambda x: x[1]['best_score'])[0],
                'average_score': sum(scores) / len(scores),
                'score_variance': sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
            }
        
        # Agent specialization analysis
        agent_specializations = {}
        for domain, results in successful_domains.items():
            if 'agent_contributions' in results:
                for agent_id, contribution in results['agent_contributions'].items():
                    if agent_id not in agent_specializations:
                        agent_specializations[agent_id] = {}
                    agent_specializations[agent_id][domain] = contribution
        
        # Find best specialist for each domain
        domain_specialists = {}
        for domain in successful_domains.keys():
            best_agent = None
            best_contribution = 0.0
            
            for agent_id, specializations in agent_specializations.items():
                if domain in specializations and specializations[domain] > best_contribution:
                    best_contribution = specializations[domain]
                    best_agent = agent_id
            
            if best_agent:
                domain_specialists[domain] = {
                    'agent_id': best_agent,
                    'contribution': best_contribution
                }
        
        analysis['specialization'] = {
            'agent_specializations': agent_specializations,
            'domain_specialists': domain_specialists
        }
        
        # Network evolution analysis
        network_stats = system.network.get_network_statistics()
        analysis['network'] = {
            'final_connections': network_stats['active_connections'],
            'network_density': network_stats['network_density'],
            'average_strength': network_stats['strength_stats']['mean']
        }
        
        # Convergence analysis
        convergence_data = {}
        for domain, results in successful_domains.items():
            if 'iteration_history' in results:
                history = results['iteration_history']
                if len(history) >= 3:
                    # Calculate convergence rate
                    final_scores = [h['best_score'] for h in history[-3:]]
                    convergence_data[domain] = {
                        'final_improvement': final_scores[-1] - final_scores[0],
                        'converged_early': len(history) < 10
                    }
        
        analysis['convergence'] = convergence_data
        
        return analysis
    
    def print_summary(self) -> None:
        """Print experiment summary."""
        if not self.results:
            self.logger.warning("No results to summarize")
            return
        
        self.logger.info("\n" + "="*60)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("="*60)
        
        # Overall statistics
        domain_results = self.results['domain_results']
        successful_domains = [d for d, r in domain_results.items() if 'best_score' in r]
        
        self.logger.info(f"Domains tested: {len(domain_results)}")
        self.logger.info(f"Successful runs: {len(successful_domains)}")
        self.logger.info(f"Total time: {self.results['experiment_metadata']['total_time']:.2f}s")
        
        # Performance by domain
        self.logger.info("\nPerformance by domain:")
        for domain in successful_domains:
            results = domain_results[domain]
            self.logger.info(f"  {domain:12}: {results['best_score']:.4f} "
                           f"({results['iterations']} iterations)")
        
        # Best performing domain
        if 'performance' in self.results['cross_domain_analysis']:
            perf = self.results['cross_domain_analysis']['performance']
            self.logger.info(f"\nBest domain: {perf['best_domain']}")
            self.logger.info(f"Average score: {perf['average_score']:.4f}")
        
        # Agent specializations
        if 'specialization' in self.results['cross_domain_analysis']:
            spec = self.results['cross_domain_analysis']['specialization']
            self.logger.info("\nDomain specialists:")
            for domain, specialist in spec['domain_specialists'].items():
                self.logger.info(f"  {domain:12}: {specialist['agent_id']} "
                               f"(contribution: {specialist['contribution']:.3f})")
        
        # Network statistics
        if 'network' in self.results['cross_domain_analysis']:
            net = self.results['cross_domain_analysis']['network']
            self.logger.info(f"\nFinal network:")
            self.logger.info(f"  Active connections: {net['final_connections']}")
            self.logger.info(f"  Network density: {net['network_density']:.3f}")
            self.logger.info(f"  Average strength: {net['average_strength']:.3f}")


def main():
    """Run multi-domain experiment."""
    
    # Configure LLM (using mock for demonstration)
    llm_config = {
        'provider': 'mock',
        'mock_responses': [
            "I'll approach this systematically with clear reasoning and step-by-step analysis.",
            "Let me break down this problem and provide a comprehensive solution with examples.",
            "Here's my detailed approach to solving this, considering multiple perspectives.",
            "I'll work through this methodically, showing all steps and explaining my reasoning.",
            "Let me provide a thorough solution with clear explanations and proper structure."
        ]
    }
    
    # Define domains and initial prompts
    domains = ['math', 'writing', 'programming']
    initial_prompts = {
        'math': 'Solve this mathematical problem with clear step-by-step reasoning:',
        'writing': 'Create an engaging and well-structured piece of writing:',
        'programming': 'Write clean, efficient code with proper documentation:'
    }
    
    # Run experiment
    experiment = MultiDomainExperiment(llm_config)
    
    try:
        results = experiment.run_experiment(
            domains=domains,
            initial_prompts=initial_prompts,
            max_iterations=12,
            population_size=10
        )
        
        # Print summary
        experiment.print_summary()
        
        return 0
        
    except Exception as e:
        experiment.logger.error(f"Experiment failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
