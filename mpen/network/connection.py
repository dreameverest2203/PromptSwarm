"""
Connection classes for the adaptive network system.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
import time


class ConnectionType(Enum):
    """Types of connections between agents."""
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    MENTORING = "mentoring"
    VALIDATION = "validation"


@dataclass
class ConnectionMetrics:
    """Metrics for tracking connection performance."""
    interaction_count: int = 0
    success_count: int = 0
    total_benefit: float = 0.0
    average_response_time: float = 0.0
    last_interaction: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of interactions."""
        if self.interaction_count == 0:
            return 0.0
        return self.success_count / self.interaction_count
    
    @property
    def average_benefit(self) -> float:
        """Calculate average benefit per interaction."""
        if self.success_count == 0:
            return 0.0
        return self.total_benefit / self.success_count


class Connection:
    """
    Represents a connection between two agents in the network.
    
    Connections adapt their strength based on collaboration success,
    track performance metrics, and can evolve over time.
    """
    
    def __init__(
        self,
        source_agent_id: str,
        target_agent_id: str,
        connection_type: ConnectionType = ConnectionType.PEER_TO_PEER,
        initial_strength: float = 0.5,
        adaptation_rate: float = 0.1
    ):
        """
        Initialize a connection between two agents.
        
        Args:
            source_agent_id: ID of the source agent
            target_agent_id: ID of the target agent
            connection_type: Type of connection
            initial_strength: Initial strength (0.0 to 1.0)
            adaptation_rate: Rate at which strength adapts
        """
        self.source_agent_id = source_agent_id
        self.target_agent_id = target_agent_id
        self.connection_type = connection_type
        self.strength = max(0.0, min(1.0, initial_strength))
        self.adaptation_rate = adaptation_rate
        
        # Performance tracking
        self.metrics = ConnectionMetrics()
        
        # Connection state
        self.is_active = True
        self.created_at = time.time()
        self.last_updated = time.time()
        
        # Domain-specific strengths
        self.domain_strengths: Dict[str, float] = {}
        
        # Interaction history
        self.interaction_history: list = []
    
    def record_interaction(
        self,
        success: bool,
        benefit: float,
        response_time: float,
        domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an interaction between the connected agents.
        
        Args:
            success: Whether the interaction was successful
            benefit: Benefit gained from the interaction (0.0 to 1.0)
            response_time: Time taken for the interaction
            domain: Domain of the interaction (optional)
            metadata: Additional metadata about the interaction
        """
        current_time = time.time()
        
        # Update metrics
        self.metrics.interaction_count += 1
        self.metrics.last_interaction = current_time
        
        if success:
            self.metrics.success_count += 1
            self.metrics.total_benefit += benefit
        
        # Update average response time
        if self.metrics.average_response_time == 0:
            self.metrics.average_response_time = response_time
        else:
            self.metrics.average_response_time = (
                0.9 * self.metrics.average_response_time + 0.1 * response_time
            )
        
        # Update domain-specific strength
        if domain:
            if domain not in self.domain_strengths:
                self.domain_strengths[domain] = 0.5
            
            domain_benefit = benefit if success else -0.1
            current_domain_strength = self.domain_strengths[domain]
            self.domain_strengths[domain] = max(0.0, min(1.0,
                current_domain_strength + self.adaptation_rate * domain_benefit
            ))
        
        # Record interaction in history
        interaction_record = {
            'timestamp': current_time,
            'success': success,
            'benefit': benefit,
            'response_time': response_time,
            'domain': domain,
            'metadata': metadata or {}
        }
        self.interaction_history.append(interaction_record)
        
        # Keep only recent history (last 100 interactions)
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]
        
        # Update connection strength
        self._update_strength(success, benefit)
        
        self.last_updated = current_time
    
    def _update_strength(self, success: bool, benefit: float) -> None:
        """Update connection strength based on interaction outcome."""
        if success:
            # Strengthen successful connections
            strength_delta = self.adaptation_rate * benefit
        else:
            # Weaken unsuccessful connections
            strength_delta = -self.adaptation_rate * 0.2
        
        # Apply update with bounds checking
        new_strength = self.strength + strength_delta
        self.strength = max(0.0, min(1.0, new_strength))
        
        # Deactivate very weak connections
        if self.strength < 0.1:
            self.is_active = False
    
    def get_strength_for_domain(self, domain: str) -> float:
        """Get connection strength for a specific domain."""
        if domain in self.domain_strengths:
            return self.domain_strengths[domain]
        return self.strength
    
    def get_recent_performance(self, window_size: int = 10) -> Dict[str, float]:
        """Get performance metrics for recent interactions."""
        recent_interactions = self.interaction_history[-window_size:]
        
        if not recent_interactions:
            return {
                'success_rate': 0.0,
                'average_benefit': 0.0,
                'interaction_count': 0
            }
        
        successes = sum(1 for i in recent_interactions if i['success'])
        total_benefit = sum(i['benefit'] for i in recent_interactions if i['success'])
        
        return {
            'success_rate': successes / len(recent_interactions),
            'average_benefit': total_benefit / successes if successes > 0 else 0.0,
            'interaction_count': len(recent_interactions)
        }
    
    def should_prune(self, inactivity_threshold: float = 3600.0) -> bool:
        """
        Determine if this connection should be pruned.
        
        Args:
            inactivity_threshold: Seconds of inactivity before considering pruning
            
        Returns:
            True if connection should be pruned
        """
        current_time = time.time()
        
        # Prune if inactive
        if not self.is_active:
            return True
        
        # Prune if no recent interactions
        if (self.metrics.last_interaction and 
            current_time - self.metrics.last_interaction > inactivity_threshold):
            return True
        
        # Prune if consistently poor performance
        if (self.metrics.interaction_count > 10 and 
            self.metrics.success_rate < 0.2):
            return True
        
        return False
    
    def get_connection_summary(self) -> Dict[str, Any]:
        """Get a summary of connection state and performance."""
        return {
            'source_agent': self.source_agent_id,
            'target_agent': self.target_agent_id,
            'connection_type': self.connection_type.value,
            'strength': self.strength,
            'is_active': self.is_active,
            'metrics': {
                'interaction_count': self.metrics.interaction_count,
                'success_rate': self.metrics.success_rate,
                'average_benefit': self.metrics.average_benefit,
                'average_response_time': self.metrics.average_response_time
            },
            'domain_strengths': self.domain_strengths.copy(),
            'age': time.time() - self.created_at,
            'last_interaction_age': (
                time.time() - self.metrics.last_interaction
                if self.metrics.last_interaction else None
            )
        }
    
    def clone_with_new_agents(
        self,
        new_source_id: str,
        new_target_id: str
    ) -> 'Connection':
        """
        Create a new connection with different agents but similar properties.
        
        Args:
            new_source_id: New source agent ID
            new_target_id: New target agent ID
            
        Returns:
            New connection instance
        """
        new_connection = Connection(
            source_agent_id=new_source_id,
            target_agent_id=new_target_id,
            connection_type=self.connection_type,
            initial_strength=self.strength,
            adaptation_rate=self.adaptation_rate
        )
        
        # Copy domain strengths
        new_connection.domain_strengths = self.domain_strengths.copy()
        
        return new_connection
    
    def __repr__(self) -> str:
        return (
            f"Connection({self.source_agent_id} -> {self.target_agent_id}, "
            f"type={self.connection_type.value}, strength={self.strength:.3f})"
        )
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Connection):
            return False
        return (
            self.source_agent_id == other.source_agent_id and
            self.target_agent_id == other.target_agent_id
        )
    
    def __hash__(self) -> int:
        return hash((self.source_agent_id, self.target_agent_id))
