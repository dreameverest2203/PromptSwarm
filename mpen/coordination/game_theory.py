"""
Game theory-based coordination for MPEN agents.

This module implements game theory mechanisms to coordinate agent interactions,
determining when agents should cooperate vs compete based on Nash equilibrium
analysis and payoff matrices.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass

from ..agents.base import BaseAgent


class StrategyType(Enum):
    """Types of coordination strategies."""
    COOPERATE = "cooperate"
    COMPETE = "compete"
    MIXED = "mixed"


@dataclass
class GameOutcome:
    """Outcome of a game theory analysis."""
    strategy: StrategyType
    expected_payoff: float
    confidence: float
    reasoning: str


class CooperationStrategy:
    """Cooperation strategy implementation."""
    
    def __init__(self, cooperation_bonus: float = 1.5):
        """
        Initialize cooperation strategy.
        
        Args:
            cooperation_bonus: Multiplier for cooperative payoffs
        """
        self.cooperation_bonus = cooperation_bonus
    
    def calculate_payoff(
        self,
        agent1_performance: float,
        agent2_performance: float,
        collaboration_history: List[Dict[str, Any]]
    ) -> float:
        """Calculate payoff for cooperation."""
        base_payoff = (agent1_performance + agent2_performance) / 2
        
        # Bonus for successful past collaborations
        if collaboration_history:
            success_rate = sum(1 for h in collaboration_history if h.get('success', False)) / len(collaboration_history)
            cooperation_bonus = self.cooperation_bonus * success_rate
        else:
            cooperation_bonus = self.cooperation_bonus * 0.5  # Default assumption
        
        return base_payoff * cooperation_bonus


class CompetitionStrategy:
    """Competition strategy implementation."""
    
    def __init__(self, winner_bonus: float = 1.8, loser_penalty: float = 0.3):
        """
        Initialize competition strategy.
        
        Args:
            winner_bonus: Multiplier for winner's payoff
            loser_penalty: Multiplier for loser's payoff
        """
        self.winner_bonus = winner_bonus
        self.loser_penalty = loser_penalty
    
    def calculate_payoff(
        self,
        agent1_performance: float,
        agent2_performance: float,
        competition_history: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """Calculate payoffs for competition."""
        if agent1_performance > agent2_performance:
            # Agent 1 wins
            agent1_payoff = agent1_performance * self.winner_bonus
            agent2_payoff = agent2_performance * self.loser_penalty
        elif agent2_performance > agent1_performance:
            # Agent 2 wins
            agent1_payoff = agent1_performance * self.loser_penalty
            agent2_payoff = agent2_performance * self.winner_bonus
        else:
            # Tie - moderate payoffs
            agent1_payoff = agent1_performance
            agent2_payoff = agent2_performance
        
        return agent1_payoff, agent2_payoff


class GameTheoryCoordinator:
    """
    Coordinates agent interactions using game theory principles.
    
    Analyzes agent performance and interaction history to determine
    optimal coordination strategies (cooperation vs competition).
    """
    
    def __init__(
        self,
        cooperation_threshold: float = 0.6,
        competition_threshold: float = 0.4,
        history_weight: float = 0.3
    ):
        """
        Initialize game theory coordinator.
        
        Args:
            cooperation_threshold: Threshold for preferring cooperation
            competition_threshold: Threshold for preferring competition
            history_weight: Weight of historical interactions in decisions
        """
        self.cooperation_threshold = cooperation_threshold
        self.competition_threshold = competition_threshold
        self.history_weight = history_weight
        
        self.cooperation_strategy = CooperationStrategy()
        self.competition_strategy = CompetitionStrategy()
        
        # Track interaction history
        self.interaction_history: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        
        self.logger = logging.getLogger("mpen.coordination.game_theory")
    
    def analyze_interaction(
        self,
        agent1: BaseAgent,
        agent2: BaseAgent,
        task_context: Optional[Dict[str, Any]] = None
    ) -> GameOutcome:
        """
        Analyze optimal interaction strategy between two agents.
        
        Args:
            agent1: First agent
            agent2: Second agent
            task_context: Context about the current task
            
        Returns:
            GameOutcome with recommended strategy
        """
        # Get agent performance metrics
        agent1_perf = self._get_agent_performance(agent1, task_context)
        agent2_perf = self._get_agent_performance(agent2, task_context)
        
        # Get interaction history
        history_key = self._get_history_key(agent1.agent_id, agent2.agent_id)
        history = self.interaction_history.get(history_key, [])
        
        # Calculate payoffs for different strategies
        coop_payoff = self._calculate_cooperation_payoff(
            agent1_perf, agent2_perf, history
        )
        comp_payoff1, comp_payoff2 = self._calculate_competition_payoffs(
            agent1_perf, agent2_perf, history
        )
        
        # Determine optimal strategy
        strategy, expected_payoff, confidence, reasoning = self._select_optimal_strategy(
            coop_payoff, comp_payoff1, comp_payoff2, agent1_perf, agent2_perf, history
        )
        
        return GameOutcome(
            strategy=strategy,
            expected_payoff=expected_payoff,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _get_agent_performance(
        self,
        agent: BaseAgent,
        task_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Get agent's current performance score."""
        # Use recent performance if available
        if hasattr(agent, 'metrics') and agent.metrics.total_calls > 0:
            success_rate = agent.metrics.successful_calls / agent.metrics.total_calls
            return success_rate
        
        # Use domain expertise if available
        if task_context and 'domain' in task_context:
            domain = task_context['domain']
            if domain in agent.domain_expertise:
                return agent.domain_expertise[domain]
        
        # Default performance
        return 0.5
    
    def _calculate_cooperation_payoff(
        self,
        agent1_perf: float,
        agent2_perf: float,
        history: List[Dict[str, Any]]
    ) -> float:
        """Calculate expected payoff from cooperation."""
        return self.cooperation_strategy.calculate_payoff(
            agent1_perf, agent2_perf, history
        )
    
    def _calculate_competition_payoffs(
        self,
        agent1_perf: float,
        agent2_perf: float,
        history: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """Calculate expected payoffs from competition."""
        return self.competition_strategy.calculate_payoff(
            agent1_perf, agent2_perf, history
        )
    
    def _select_optimal_strategy(
        self,
        coop_payoff: float,
        comp_payoff1: float,
        comp_payoff2: float,
        agent1_perf: float,
        agent2_perf: float,
        history: List[Dict[str, Any]]
    ) -> Tuple[StrategyType, float, float, str]:
        """Select optimal strategy based on payoff analysis."""
        
        # Average competition payoff
        avg_comp_payoff = (comp_payoff1 + comp_payoff2) / 2
        
        # Factor in historical success
        if history:
            coop_success_rate = sum(
                1 for h in history 
                if h.get('strategy') == 'cooperate' and h.get('success', False)
            ) / max(1, sum(1 for h in history if h.get('strategy') == 'cooperate'))
            
            comp_success_rate = sum(
                1 for h in history 
                if h.get('strategy') == 'compete' and h.get('success', False)
            ) / max(1, sum(1 for h in history if h.get('strategy') == 'compete'))
            
            # Adjust payoffs based on historical success
            coop_payoff *= (1 + self.history_weight * coop_success_rate)
            avg_comp_payoff *= (1 + self.history_weight * comp_success_rate)
        
        # Decision logic
        payoff_difference = coop_payoff - avg_comp_payoff
        
        if payoff_difference > self.cooperation_threshold:
            return (
                StrategyType.COOPERATE,
                coop_payoff,
                min(1.0, payoff_difference),
                f"Cooperation expected payoff ({coop_payoff:.3f}) significantly exceeds competition ({avg_comp_payoff:.3f})"
            )
        elif payoff_difference < -self.competition_threshold:
            return (
                StrategyType.COMPETE,
                avg_comp_payoff,
                min(1.0, -payoff_difference),
                f"Competition expected payoff ({avg_comp_payoff:.3f}) significantly exceeds cooperation ({coop_payoff:.3f})"
            )
        else:
            # Mixed strategy
            coop_prob = 0.5 + payoff_difference / 2.0
            mixed_payoff = coop_prob * coop_payoff + (1 - coop_prob) * avg_comp_payoff
            
            return (
                StrategyType.MIXED,
                mixed_payoff,
                0.5,
                f"Mixed strategy with {coop_prob:.1%} cooperation probability"
            )
    
    def record_interaction_outcome(
        self,
        agent1_id: str,
        agent2_id: str,
        strategy: StrategyType,
        success: bool,
        payoff: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record the outcome of an agent interaction."""
        history_key = self._get_history_key(agent1_id, agent2_id)
        
        if history_key not in self.interaction_history:
            self.interaction_history[history_key] = []
        
        outcome_record = {
            'strategy': strategy.value,
            'success': success,
            'payoff': payoff,
            'timestamp': np.datetime64('now'),
            'metadata': metadata or {}
        }
        
        self.interaction_history[history_key].append(outcome_record)
        
        # Keep limited history
        if len(self.interaction_history[history_key]) > 50:
            self.interaction_history[history_key] = self.interaction_history[history_key][-50:]
    
    def _get_history_key(self, agent1_id: str, agent2_id: str) -> Tuple[str, str]:
        """Get normalized history key for agent pair."""
        return tuple(sorted([agent1_id, agent2_id]))
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get statistics about coordination decisions."""
        if not self.interaction_history:
            return {'message': 'No interaction history available'}
        
        total_interactions = sum(len(history) for history in self.interaction_history.values())
        
        strategy_counts = {'cooperate': 0, 'compete': 0, 'mixed': 0}
        success_by_strategy = {'cooperate': [], 'compete': [], 'mixed': []}
        
        for history in self.interaction_history.values():
            for record in history:
                strategy = record['strategy']
                if strategy in strategy_counts:
                    strategy_counts[strategy] += 1
                    success_by_strategy[strategy].append(record['success'])
        
        # Calculate success rates
        success_rates = {}
        for strategy, successes in success_by_strategy.items():
            if successes:
                success_rates[strategy] = sum(successes) / len(successes)
            else:
                success_rates[strategy] = 0.0
        
        return {
            'total_interactions': total_interactions,
            'strategy_distribution': strategy_counts,
            'success_rates': success_rates,
            'unique_agent_pairs': len(self.interaction_history),
            'average_history_length': total_interactions / len(self.interaction_history)
        }
    
    def analyze_nash_equilibrium(
        self,
        agents: List[BaseAgent],
        task_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze Nash equilibrium for multi-agent coordination.
        
        Args:
            agents: List of agents to analyze
            task_context: Context about current task
            
        Returns:
            Nash equilibrium analysis
        """
        n_agents = len(agents)
        
        if n_agents < 2:
            return {'error': 'Need at least 2 agents for equilibrium analysis'}
        
        # Create payoff matrix for all agent pairs
        payoff_matrix = np.zeros((n_agents, n_agents, 2, 2))  # [agent1, agent2, strategy1, strategy2]
        
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    # Calculate payoffs for different strategy combinations
                    agent1_perf = self._get_agent_performance(agent1, task_context)
                    agent2_perf = self._get_agent_performance(agent2, task_context)
                    
                    history_key = self._get_history_key(agent1.agent_id, agent2.agent_id)
                    history = self.interaction_history.get(history_key, [])
                    
                    # Cooperate-Cooperate
                    coop_payoff = self._calculate_cooperation_payoff(agent1_perf, agent2_perf, history)
                    payoff_matrix[i, j, 0, 0] = coop_payoff
                    payoff_matrix[j, i, 0, 0] = coop_payoff
                    
                    # Compete-Compete
                    comp_payoff1, comp_payoff2 = self._calculate_competition_payoffs(agent1_perf, agent2_perf, history)
                    payoff_matrix[i, j, 1, 1] = comp_payoff1
                    payoff_matrix[j, i, 1, 1] = comp_payoff2
                    
                    # Mixed strategies (simplified)
                    payoff_matrix[i, j, 0, 1] = agent1_perf * 0.8  # Cooperate vs Compete
                    payoff_matrix[i, j, 1, 0] = agent1_perf * 1.2  # Compete vs Cooperate
                    payoff_matrix[j, i, 0, 1] = agent2_perf * 1.2
                    payoff_matrix[j, i, 1, 0] = agent2_perf * 0.8
        
        # Find Nash equilibrium (simplified analysis)
        equilibrium_strategies = self._find_nash_equilibrium(payoff_matrix, agents)
        
        return {
            'equilibrium_strategies': equilibrium_strategies,
            'payoff_matrix_shape': payoff_matrix.shape,
            'agents_analyzed': [agent.agent_id for agent in agents]
        }
    
    def _find_nash_equilibrium(
        self,
        payoff_matrix: np.ndarray,
        agents: List[BaseAgent]
    ) -> Dict[str, str]:
        """Find Nash equilibrium strategies (simplified implementation)."""
        n_agents = len(agents)
        strategies = {}
        
        # Simplified Nash equilibrium finding
        # In practice, this would use more sophisticated game theory algorithms
        
        for i, agent in enumerate(agents):
            # Find best response to other agents' strategies
            cooperate_payoff = 0.0
            compete_payoff = 0.0
            
            for j in range(n_agents):
                if i != j:
                    cooperate_payoff += payoff_matrix[i, j, 0, 0]  # Assume others cooperate
                    compete_payoff += payoff_matrix[i, j, 1, 1]   # Assume others compete
            
            if cooperate_payoff > compete_payoff:
                strategies[agent.agent_id] = 'cooperate'
            elif compete_payoff > cooperate_payoff:
                strategies[agent.agent_id] = 'compete'
            else:
                strategies[agent.agent_id] = 'mixed'
        
        return strategies
    
    def recommend_coordination_policy(
        self,
        agents: List[BaseAgent],
        task_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Recommend coordination policy for a group of agents.
        
        Args:
            agents: List of agents
            task_context: Task context
            
        Returns:
            Coordination policy recommendations
        """
        if len(agents) < 2:
            return {'policy': 'single_agent', 'reasoning': 'Only one agent available'}
        
        # Analyze pairwise interactions
        pairwise_analyses = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                analysis = self.analyze_interaction(agents[i], agents[j], task_context)
                pairwise_analyses.append({
                    'agents': (agents[i].agent_id, agents[j].agent_id),
                    'analysis': analysis
                })
        
        # Aggregate recommendations
        strategy_votes = {'cooperate': 0, 'compete': 0, 'mixed': 0}
        total_confidence = 0.0
        
        for analysis in pairwise_analyses:
            strategy = analysis['analysis'].strategy.value
            confidence = analysis['analysis'].confidence
            
            strategy_votes[strategy] += confidence
            total_confidence += confidence
        
        # Determine overall policy
        if total_confidence > 0:
            # Normalize votes
            for strategy in strategy_votes:
                strategy_votes[strategy] /= total_confidence
            
            best_strategy = max(strategy_votes.items(), key=lambda x: x[1])[0]
        else:
            best_strategy = 'mixed'
        
        return {
            'policy': best_strategy,
            'strategy_scores': strategy_votes,
            'confidence': total_confidence / len(pairwise_analyses) if pairwise_analyses else 0.0,
            'pairwise_analyses': len(pairwise_analyses),
            'reasoning': f"Based on {len(pairwise_analyses)} pairwise analyses, {best_strategy} strategy is optimal"
        }
