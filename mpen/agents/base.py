"""
Base agent class for all MPEN agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, field
import time

from ..utils.llm_interface import LLMInterface


@dataclass
class AgentMetrics:
    """Metrics tracking for agent performance."""
    total_calls: int = 0
    successful_calls: int = 0
    average_response_time: float = 0.0
    collaboration_success_rate: Dict[str, float] = field(default_factory=dict)
    specialization_scores: Dict[str, float] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base class for all MPEN agents.
    
    Provides common functionality for agent communication, metrics tracking,
    and specialization development.
    """
    
    def __init__(
        self, 
        agent_id: str, 
        llm_config: Optional[Dict[str, Any]] = None,
        specialization_domains: Optional[List[str]] = None
    ):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            llm_config: Configuration for language model interface
            specialization_domains: Domains this agent can specialize in
        """
        self.agent_id = agent_id
        self.agent_type = self.__class__.__name__
        self.llm = LLMInterface(llm_config or {})
        self.specialization_domains = specialization_domains or []
        
        # Performance tracking
        self.metrics = AgentMetrics()
        self.collaboration_history: List[Dict[str, Any]] = []
        
        # Specialization tracking
        self.domain_expertise: Dict[str, float] = {
            domain: 0.5 for domain in self.specialization_domains
        }
        
        self.logger = logging.getLogger(f"mpen.{self.agent_type}.{agent_id}")
        self.logger.info(f"Initialized {self.agent_type} agent: {agent_id}")
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for the agent.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Dictionary containing agent's output and metadata
        """
        pass
    
    def collaborate(
        self, 
        other_agent: 'BaseAgent', 
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Collaborate with another agent on a task.
        
        Args:
            other_agent: The agent to collaborate with
            task_data: Data about the collaborative task
            
        Returns:
            Results of the collaboration
        """
        start_time = time.time()
        
        try:
            # Record collaboration attempt
            collaboration_record = {
                'partner_id': other_agent.agent_id,
                'partner_type': other_agent.agent_type,
                'task_type': task_data.get('type', 'unknown'),
                'timestamp': start_time
            }
            
            # Perform collaboration logic (to be implemented by subclasses)
            result = self._collaborate_impl(other_agent, task_data)
            
            # Record success
            collaboration_record['success'] = True
            collaboration_record['duration'] = time.time() - start_time
            collaboration_record['result_quality'] = result.get('quality', 0.5)
            
            self.collaboration_history.append(collaboration_record)
            
            # Update collaboration success rates
            partner_key = f"{other_agent.agent_type}_{other_agent.agent_id}"
            if partner_key not in self.metrics.collaboration_success_rate:
                self.metrics.collaboration_success_rate[partner_key] = 0.0
            
            # Exponential moving average for success rate
            current_rate = self.metrics.collaboration_success_rate[partner_key]
            self.metrics.collaboration_success_rate[partner_key] = (
                0.9 * current_rate + 0.1 * 1.0
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Collaboration failed with {other_agent.agent_id}: {e}")
            
            # Record failure
            collaboration_record['success'] = False
            collaboration_record['error'] = str(e)
            self.collaboration_history.append(collaboration_record)
            
            return {'success': False, 'error': str(e)}
    
    def _collaborate_impl(
        self, 
        other_agent: 'BaseAgent', 
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Implementation of collaboration logic.
        Override in subclasses for specific collaboration patterns.
        """
        return {
            'success': True,
            'message': f"Basic collaboration between {self.agent_id} and {other_agent.agent_id}",
            'quality': 0.5
        }
    
    def update_specialization(
        self, 
        domain: str, 
        performance_score: float
    ) -> None:
        """
        Update specialization scores based on performance.
        
        Args:
            domain: Domain to update specialization for
            performance_score: Score between 0 and 1
        """
        if domain not in self.domain_expertise:
            self.domain_expertise[domain] = 0.5
        
        # Exponential moving average for specialization
        current_expertise = self.domain_expertise[domain]
        self.domain_expertise[domain] = (
            0.8 * current_expertise + 0.2 * performance_score
        )
        
        self.logger.debug(
            f"Updated {domain} expertise to {self.domain_expertise[domain]:.3f}"
        )
    
    def get_best_specialization(self) -> Optional[str]:
        """Get the domain this agent is most specialized in."""
        if not self.domain_expertise:
            return None
        
        return max(self.domain_expertise.items(), key=lambda x: x[1])[0]
    
    def get_collaboration_preference(
        self, 
        agent_type: str
    ) -> float:
        """
        Get preference score for collaborating with a specific agent type.
        
        Args:
            agent_type: Type of agent to get preference for
            
        Returns:
            Preference score between 0 and 1
        """
        # Calculate average success rate with this agent type
        relevant_rates = [
            rate for key, rate in self.metrics.collaboration_success_rate.items()
            if key.startswith(agent_type)
        ]
        
        if not relevant_rates:
            return 0.5  # Neutral preference for new agent types
        
        return sum(relevant_rates) / len(relevant_rates)
    
    def update_metrics(
        self, 
        call_successful: bool, 
        response_time: float
    ) -> None:
        """Update agent performance metrics."""
        self.metrics.total_calls += 1
        
        if call_successful:
            self.metrics.successful_calls += 1
        
        # Update average response time with exponential moving average
        if self.metrics.average_response_time == 0:
            self.metrics.average_response_time = response_time
        else:
            self.metrics.average_response_time = (
                0.9 * self.metrics.average_response_time + 0.1 * response_time
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of agent performance."""
        success_rate = (
            self.metrics.successful_calls / self.metrics.total_calls
            if self.metrics.total_calls > 0 else 0.0
        )
        
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'success_rate': success_rate,
            'total_calls': self.metrics.total_calls,
            'avg_response_time': self.metrics.average_response_time,
            'best_specialization': self.get_best_specialization(),
            'domain_expertise': self.domain_expertise.copy(),
            'collaboration_rates': self.metrics.collaboration_success_rate.copy()
        }
    
    def __repr__(self) -> str:
        return f"{self.agent_type}(id={self.agent_id})"
