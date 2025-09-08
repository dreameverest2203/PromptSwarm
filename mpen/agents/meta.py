"""
Meta Agent for coordinating the network and managing global optimization.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

from .base import BaseAgent
from ..tasks.base import Task


@dataclass
class CoordinationDecision:
    """Represents a coordination decision made by the meta agent."""
    action: str  # 'collaborate', 'compete', 'specialize', 'explore'
    agents: List[str]  # Agent IDs involved
    rationale: str
    expected_benefit: float
    priority: float


class MetaAgent(BaseAgent):
    """
    Meta Agent coordinates the overall optimization process and agent interactions.
    
    Makes strategic decisions about agent collaboration, competition, specialization,
    and exploration based on game theory principles and system performance.
    """
    
    def __init__(
        self, 
        agent_id: str, 
        llm_config: Optional[Dict[str, Any]] = None,
        coordination_strategies: Optional[List[str]] = None
    ):
        """
        Initialize Meta Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            llm_config: Configuration for language model interface
            coordination_strategies: Strategies for coordinating other agents
        """
        specialization_domains = [
            'network_coordination',
            'strategic_planning',
            'resource_allocation',
            'performance_optimization',
            'conflict_resolution',
            'exploration_management'
        ]
        
        super().__init__(agent_id, llm_config, specialization_domains)
        
        self.coordination_strategies = coordination_strategies or [
            'cooperative_optimization',
            'competitive_selection',
            'specialization_guidance',
            'exploration_exploitation',
            'conflict_resolution',
            'resource_balancing'
        ]
        
        # Game theory parameters
        self.cooperation_threshold = 0.7
        self.competition_threshold = 0.3
        self.exploration_rate = 0.2
        
        # System state tracking
        self.system_performance_history: List[Dict[str, Any]] = []
        self.agent_performance_tracking: Dict[str, List[float]] = {}
        self.coordination_history: List[CoordinationDecision] = []
        
        # Strategic knowledge
        self.successful_collaborations: Dict[Tuple[str, str], float] = {}
        self.domain_specialists: Dict[str, List[str]] = {}
        
        self.logger = logging.getLogger(f"mpen.MetaAgent.{agent_id}")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process coordination requests and make strategic decisions.
        
        Args:
            input_data: Contains system state, agent performances, and coordination needs
            
        Returns:
            Coordination decisions and strategic guidance
        """
        system_state = input_data.get('system_state', {})
        agent_performances = input_data.get('agent_performances', {})
        current_task = input_data.get('current_task')
        coordination_request = input_data.get('coordination_request')
        
        # Update internal state
        self._update_system_knowledge(system_state, agent_performances)
        
        # Make coordination decisions
        decisions = self._make_coordination_decisions(
            system_state, agent_performances, current_task, coordination_request
        )
        
        # Record decisions
        for decision in decisions:
            self.coordination_history.append(decision)
        
        return {
            'coordination_decisions': [
                {
                    'action': d.action,
                    'agents': d.agents,
                    'rationale': d.rationale,
                    'priority': d.priority
                }
                for d in decisions
            ],
            'system_recommendations': self._generate_system_recommendations(system_state),
            'meta_agent_id': self.agent_id
        }
    
    def coordinate_agents(
        self,
        available_agents: List[BaseAgent],
        task: Task,
        system_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate agents for optimal task performance.
        
        Args:
            available_agents: List of agents available for coordination
            task: Current task being optimized
            system_state: Current state of the system
            
        Returns:
            Coordination plan with agent assignments and strategies
        """
        agent_performances = {
            agent.agent_id: self._evaluate_agent_for_task(agent, task)
            for agent in available_agents
        }
        
        # Determine coordination strategy
        strategy = self._select_coordination_strategy(task, agent_performances, system_state)
        
        if strategy == 'cooperative_optimization':
            return self._plan_cooperative_optimization(available_agents, task, agent_performances)
        elif strategy == 'competitive_selection':
            return self._plan_competitive_selection(available_agents, task, agent_performances)
        elif strategy == 'specialization_guidance':
            return self._plan_specialization_guidance(available_agents, task, agent_performances)
        elif strategy == 'exploration_exploitation':
            return self._plan_exploration_exploitation(available_agents, task, agent_performances)
        else:
            return self._plan_default_coordination(available_agents, task, agent_performances)
    
    def _update_system_knowledge(
        self,
        system_state: Dict[str, Any],
        agent_performances: Dict[str, Any]
    ) -> None:
        """Update internal knowledge about system and agent performance."""
        # Track system performance
        if 'current_score' in system_state:
            performance_record = {
                'iteration': system_state.get('iteration', 0),
                'score': system_state['current_score'],
                'agent_count': len(agent_performances),
                'timestamp': system_state.get('timestamp')
            }
            self.system_performance_history.append(performance_record)
        
        # Track individual agent performances
        for agent_id, performance in agent_performances.items():
            if agent_id not in self.agent_performance_tracking:
                self.agent_performance_tracking[agent_id] = []
            
            if isinstance(performance, dict) and 'score' in performance:
                self.agent_performance_tracking[agent_id].append(performance['score'])
            elif isinstance(performance, (int, float)):
                self.agent_performance_tracking[agent_id].append(performance)
        
        # Update domain specialists tracking
        for agent_id, performance in agent_performances.items():
            if isinstance(performance, dict) and 'domain' in performance:
                domain = performance['domain']
                if domain not in self.domain_specialists:
                    self.domain_specialists[domain] = []
                
                # Add to specialists if performing well
                if performance.get('score', 0) > 0.7:
                    if agent_id not in self.domain_specialists[domain]:
                        self.domain_specialists[domain].append(agent_id)
    
    def _make_coordination_decisions(
        self,
        system_state: Dict[str, Any],
        agent_performances: Dict[str, Any],
        current_task: Optional[Task],
        coordination_request: Optional[str]
    ) -> List[CoordinationDecision]:
        """Make strategic coordination decisions based on current state."""
        decisions = []
        
        # Analyze system state
        is_stagnating = self._detect_stagnation()
        is_converging = self._detect_convergence()
        performance_variance = self._calculate_performance_variance(agent_performances)
        
        # Decision 1: Collaboration vs Competition
        if performance_variance < 0.1 and not is_stagnating:
            # Agents performing similarly - encourage collaboration
            decisions.append(CoordinationDecision(
                action='collaborate',
                agents=list(agent_performances.keys()),
                rationale='Similar performance levels suggest collaborative potential',
                expected_benefit=0.8,
                priority=0.7
            ))
        elif performance_variance > 0.3:
            # High variance - use competition to drive improvement
            decisions.append(CoordinationDecision(
                action='compete',
                agents=self._select_competitive_pairs(agent_performances),
                rationale='High performance variance suggests competitive selection needed',
                expected_benefit=0.6,
                priority=0.8
            ))
        
        # Decision 2: Specialization
        if current_task and hasattr(current_task, 'domain'):
            domain = current_task.domain
            if domain not in self.domain_specialists or len(self.domain_specialists[domain]) < 2:
                # Need more specialists in this domain
                candidates = self._identify_specialization_candidates(domain, agent_performances)
                if candidates:
                    decisions.append(CoordinationDecision(
                        action='specialize',
                        agents=candidates,
                        rationale=f'Need more specialists in {domain}',
                        expected_benefit=0.7,
                        priority=0.6
                    ))
        
        # Decision 3: Exploration vs Exploitation
        if is_stagnating:
            # Encourage exploration
            decisions.append(CoordinationDecision(
                action='explore',
                agents=self._select_explorers(agent_performances),
                rationale='System stagnation detected, need exploration',
                expected_benefit=0.5,
                priority=0.9
            ))
        elif is_converging and len(self.system_performance_history) > 10:
            # Focus on exploitation
            best_performers = self._identify_best_performers(agent_performances)
            decisions.append(CoordinationDecision(
                action='exploit',
                agents=best_performers,
                rationale='System converging, focus on best strategies',
                expected_benefit=0.8,
                priority=0.7
            ))
        
        # Sort decisions by priority
        decisions.sort(key=lambda x: x.priority, reverse=True)
        
        return decisions[:3]  # Return top 3 decisions
    
    def _select_coordination_strategy(
        self,
        task: Task,
        agent_performances: Dict[str, float],
        system_state: Dict[str, Any]
    ) -> str:
        """Select the best coordination strategy for the current situation."""
        # Analyze system state
        performance_variance = np.var(list(agent_performances.values()))
        average_performance = np.mean(list(agent_performances.values()))
        
        # Check for stagnation
        is_stagnating = self._detect_stagnation()
        
        # Decision logic based on game theory principles
        if is_stagnating:
            return 'exploration_exploitation'
        elif performance_variance < 0.05:
            # Low variance - agents similar, cooperation beneficial
            return 'cooperative_optimization'
        elif performance_variance > 0.2:
            # High variance - use competition
            return 'competitive_selection'
        elif hasattr(task, 'domain') and task.domain in self.domain_specialists:
            # Have domain knowledge - use specialization
            return 'specialization_guidance'
        else:
            return 'cooperative_optimization'
    
    def _plan_cooperative_optimization(
        self,
        agents: List[BaseAgent],
        task: Task,
        performances: Dict[str, float]
    ) -> Dict[str, Any]:
        """Plan cooperative optimization strategy."""
        # Form collaborative groups
        groups = self._form_collaborative_groups(agents, task, performances)
        
        return {
            'strategy': 'cooperative_optimization',
            'groups': groups,
            'coordination_type': 'collaborative',
            'expected_synergy': 0.8
        }
    
    def _plan_competitive_selection(
        self,
        agents: List[BaseAgent],
        task: Task,
        performances: Dict[str, float]
    ) -> Dict[str, Any]:
        """Plan competitive selection strategy."""
        # Create competitive pairs
        competitive_pairs = self._create_competitive_pairs(agents, performances)
        
        return {
            'strategy': 'competitive_selection',
            'pairs': competitive_pairs,
            'coordination_type': 'competitive',
            'selection_pressure': 0.7
        }
    
    def _plan_specialization_guidance(
        self,
        agents: List[BaseAgent],
        task: Task,
        performances: Dict[str, float]
    ) -> Dict[str, Any]:
        """Plan specialization guidance strategy."""
        # Assign specialization roles
        specialization_assignments = self._assign_specializations(agents, task, performances)
        
        return {
            'strategy': 'specialization_guidance',
            'assignments': specialization_assignments,
            'coordination_type': 'specialized',
            'specialization_depth': 0.8
        }
    
    def _plan_exploration_exploitation(
        self,
        agents: List[BaseAgent],
        task: Task,
        performances: Dict[str, float]
    ) -> Dict[str, Any]:
        """Plan exploration-exploitation strategy."""
        # Balance exploration and exploitation
        explorers = self._select_explorers(performances)
        exploiters = self._select_exploiters(performances)
        
        return {
            'strategy': 'exploration_exploitation',
            'explorers': explorers,
            'exploiters': exploiters,
            'coordination_type': 'balanced',
            'exploration_rate': self.exploration_rate
        }
    
    def _plan_default_coordination(
        self,
        agents: List[BaseAgent],
        task: Task,
        performances: Dict[str, float]
    ) -> Dict[str, Any]:
        """Plan default coordination when no specific strategy is selected."""
        return {
            'strategy': 'default',
            'all_agents': [agent.agent_id for agent in agents],
            'coordination_type': 'general',
            'approach': 'balanced_participation'
        }
    
    def _evaluate_agent_for_task(self, agent: BaseAgent, task: Task) -> float:
        """Evaluate how well an agent is suited for a specific task."""
        base_score = 0.5
        
        # Check domain expertise
        if hasattr(task, 'domain') and task.domain in agent.domain_expertise:
            base_score += agent.domain_expertise[task.domain] * 0.3
        
        # Check recent performance
        if agent.agent_id in self.agent_performance_tracking:
            recent_scores = self.agent_performance_tracking[agent.agent_id][-5:]
            if recent_scores:
                base_score += np.mean(recent_scores) * 0.2
        
        return min(1.0, base_score)
    
    def _detect_stagnation(self) -> bool:
        """Detect if the system is stagnating."""
        if len(self.system_performance_history) < 5:
            return False
        
        recent_scores = [h['score'] for h in self.system_performance_history[-5:]]
        score_variance = np.var(recent_scores)
        
        return score_variance < 0.001  # Very low variance indicates stagnation
    
    def _detect_convergence(self) -> bool:
        """Detect if the system is converging."""
        if len(self.system_performance_history) < 10:
            return False
        
        recent_scores = [h['score'] for h in self.system_performance_history[-10:]]
        
        # Check if scores are consistently improving
        improvements = [
            recent_scores[i] > recent_scores[i-1] 
            for i in range(1, len(recent_scores))
        ]
        
        return sum(improvements) / len(improvements) > 0.7
    
    def _calculate_performance_variance(self, agent_performances: Dict[str, Any]) -> float:
        """Calculate variance in agent performances."""
        scores = []
        for performance in agent_performances.values():
            if isinstance(performance, dict) and 'score' in performance:
                scores.append(performance['score'])
            elif isinstance(performance, (int, float)):
                scores.append(performance)
        
        return np.var(scores) if scores else 0.0
    
    def _select_competitive_pairs(self, agent_performances: Dict[str, Any]) -> List[str]:
        """Select agents for competitive pairing."""
        # Sort agents by performance
        sorted_agents = sorted(
            agent_performances.items(),
            key=lambda x: x[1] if isinstance(x[1], (int, float)) else x[1].get('score', 0),
            reverse=True
        )
        
        # Return middle-performing agents for competition
        mid_start = len(sorted_agents) // 4
        mid_end = 3 * len(sorted_agents) // 4
        
        return [agent_id for agent_id, _ in sorted_agents[mid_start:mid_end]]
    
    def _identify_specialization_candidates(
        self,
        domain: str,
        agent_performances: Dict[str, Any]
    ) -> List[str]:
        """Identify agents that could specialize in a domain."""
        candidates = []
        
        for agent_id, performance in agent_performances.items():
            if isinstance(performance, dict):
                # Check if agent has shown aptitude in this domain
                domain_score = performance.get(f'{domain}_score', 0.5)
                if domain_score > 0.6:
                    candidates.append(agent_id)
        
        return candidates[:2]  # Limit to 2 candidates
    
    def _select_explorers(self, agent_performances: Dict[str, Any]) -> List[str]:
        """Select agents for exploration tasks."""
        # Choose agents with diverse performance patterns
        explorers = []
        
        for agent_id in agent_performances.keys():
            if agent_id in self.agent_performance_tracking:
                scores = self.agent_performance_tracking[agent_id]
                if len(scores) > 3:
                    variance = np.var(scores[-5:])
                    if variance > 0.1:  # High variance suggests exploration potential
                        explorers.append(agent_id)
        
        return explorers[:2]  # Limit to 2 explorers
    
    def _identify_best_performers(self, agent_performances: Dict[str, Any]) -> List[str]:
        """Identify the best performing agents."""
        sorted_agents = sorted(
            agent_performances.items(),
            key=lambda x: x[1] if isinstance(x[1], (int, float)) else x[1].get('score', 0),
            reverse=True
        )
        
        return [agent_id for agent_id, _ in sorted_agents[:3]]
    
    def _select_exploiters(self, agent_performances: Dict[str, Any]) -> List[str]:
        """Select agents for exploitation (consistent high performers)."""
        return self._identify_best_performers(agent_performances)
    
    def _form_collaborative_groups(
        self,
        agents: List[BaseAgent],
        task: Task,
        performances: Dict[str, float]
    ) -> List[List[str]]:
        """Form collaborative groups of agents."""
        # Simple grouping by complementary strengths
        generator_agents = [a for a in agents if a.agent_type == 'GeneratorAgent']
        critic_agents = [a for a in agents if a.agent_type == 'CriticAgent']
        validator_agents = [a for a in agents if a.agent_type == 'ValidatorAgent']
        
        groups = []
        
        # Form mixed groups with different agent types
        for i in range(min(len(generator_agents), len(critic_agents))):
            group = [generator_agents[i].agent_id, critic_agents[i].agent_id]
            if i < len(validator_agents):
                group.append(validator_agents[i].agent_id)
            groups.append(group)
        
        return groups
    
    def _create_competitive_pairs(
        self,
        agents: List[BaseAgent],
        performances: Dict[str, float]
    ) -> List[Tuple[str, str]]:
        """Create competitive pairs of agents."""
        # Pair agents of similar types for competition
        same_type_agents = {}
        for agent in agents:
            agent_type = agent.agent_type
            if agent_type not in same_type_agents:
                same_type_agents[agent_type] = []
            same_type_agents[agent_type].append(agent.agent_id)
        
        pairs = []
        for agent_type, agent_ids in same_type_agents.items():
            for i in range(0, len(agent_ids) - 1, 2):
                pairs.append((agent_ids[i], agent_ids[i + 1]))
        
        return pairs
    
    def _assign_specializations(
        self,
        agents: List[BaseAgent],
        task: Task,
        performances: Dict[str, float]
    ) -> Dict[str, str]:
        """Assign specialization domains to agents."""
        assignments = {}
        
        available_domains = ['mathematical_reasoning', 'creative_writing', 'programming']
        
        # Assign based on current performance and expertise
        for i, agent in enumerate(agents):
            domain = available_domains[i % len(available_domains)]
            assignments[agent.agent_id] = domain
        
        return assignments
    
    def _generate_system_recommendations(
        self,
        system_state: Dict[str, Any]
    ) -> List[str]:
        """Generate high-level system recommendations."""
        recommendations = []
        
        # Analyze recent performance trends
        if len(self.system_performance_history) > 5:
            recent_trend = self._analyze_performance_trend()
            
            if recent_trend == 'declining':
                recommendations.append("Consider increasing exploration rate")
                recommendations.append("Review agent specializations")
            elif recent_trend == 'stagnating':
                recommendations.append("Introduce new mutation strategies")
                recommendations.append("Increase agent diversity")
            elif recent_trend == 'improving':
                recommendations.append("Maintain current coordination strategy")
                recommendations.append("Fine-tune successful collaborations")
        
        # Resource allocation recommendations
        if 'resource_usage' in system_state:
            usage = system_state['resource_usage']
            if usage > 0.8:
                recommendations.append("Optimize agent coordination to reduce resource usage")
            elif usage < 0.3:
                recommendations.append("Consider adding more agents or increasing parallelism")
        
        return recommendations
    
    def _analyze_performance_trend(self) -> str:
        """Analyze recent performance trend."""
        if len(self.system_performance_history) < 5:
            return 'insufficient_data'
        
        recent_scores = [h['score'] for h in self.system_performance_history[-5:]]
        
        # Calculate trend
        x = np.arange(len(recent_scores))
        slope = np.polyfit(x, recent_scores, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stagnating'
    
    def get_coordination_summary(self) -> Dict[str, Any]:
        """Get summary of coordination decisions and effectiveness."""
        if not self.coordination_history:
            return {'message': 'No coordination decisions made yet'}
        
        # Analyze decision patterns
        decision_counts = {}
        for decision in self.coordination_history:
            action = decision.action
            decision_counts[action] = decision_counts.get(action, 0) + 1
        
        return {
            'total_decisions': len(self.coordination_history),
            'decision_distribution': decision_counts,
            'successful_collaborations': len(self.successful_collaborations),
            'domain_specialists': {
                domain: len(specialists) 
                for domain, specialists in self.domain_specialists.items()
            },
            'coordination_strategies': self.coordination_strategies
        }
