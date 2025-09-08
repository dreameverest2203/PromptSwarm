"""
Adaptive network system for managing agent connections and interactions.
"""

import logging
import random
import time
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

from .connection import Connection, ConnectionType
from ..agents.base import BaseAgent


class AdaptiveNetwork:
    """
    Manages the adaptive network of agent connections.
    
    The network evolves over time, strengthening successful collaborations
    and pruning ineffective connections. Supports different connection types
    and domain-specific relationship tracking.
    """
    
    def __init__(
        self,
        agents: List[BaseAgent],
        adaptation_rate: float = 0.1,
        pruning_interval: int = 50,
        max_connections_per_agent: int = 10
    ):
        """
        Initialize the adaptive network.
        
        Args:
            agents: List of agents in the network
            adaptation_rate: Rate at which connections adapt
            pruning_interval: Iterations between network pruning
            max_connections_per_agent: Maximum connections per agent
        """
        self.agents = {agent.agent_id: agent for agent in agents}
        self.adaptation_rate = adaptation_rate
        self.pruning_interval = pruning_interval
        self.max_connections_per_agent = max_connections_per_agent
        
        # Network state
        self.connections: Dict[Tuple[str, str], Connection] = {}
        self.agent_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Network evolution tracking
        self.iteration_count = 0
        self.evolution_history: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, float]] = []
        
        self.logger = logging.getLogger("mpen.network")
        
        # Initialize network with basic connections
        self._initialize_network()
        
        self.logger.info(f"Initialized adaptive network with {len(self.agents)} agents")
    
    def _initialize_network(self) -> None:
        """Initialize the network with basic connections between agents."""
        agent_ids = list(self.agents.keys())
        
        # Create connections between different agent types
        generators = [aid for aid, agent in self.agents.items() if agent.agent_type == 'GeneratorAgent']
        critics = [aid for aid, agent in self.agents.items() if agent.agent_type == 'CriticAgent']
        validators = [aid for aid, agent in self.agents.items() if agent.agent_type == 'ValidatorAgent']
        meta_agents = [aid for aid, agent in self.agents.items() if agent.agent_type == 'MetaAgent']
        
        # Generator -> Critic connections (collaborative)
        for gen_id in generators:
            for critic_id in critics:
                self._add_connection(
                    gen_id, critic_id, 
                    ConnectionType.COLLABORATIVE, 
                    initial_strength=0.7
                )
        
        # Critic -> Validator connections (validation)
        for critic_id in critics:
            for val_id in validators:
                self._add_connection(
                    critic_id, val_id,
                    ConnectionType.VALIDATION,
                    initial_strength=0.6
                )
        
        # Meta agents -> all others (hierarchical)
        for meta_id in meta_agents:
            for agent_id in agent_ids:
                if agent_id != meta_id:
                    self._add_connection(
                        meta_id, agent_id,
                        ConnectionType.HIERARCHICAL,
                        initial_strength=0.8
                    )
        
        # Some peer-to-peer connections within same types
        for agent_type_ids in [generators, critics, validators]:
            if len(agent_type_ids) > 1:
                for i in range(len(agent_type_ids) - 1):
                    self._add_connection(
                        agent_type_ids[i], agent_type_ids[i + 1],
                        ConnectionType.PEER_TO_PEER,
                        initial_strength=0.5
                    )
    
    def _add_connection(
        self,
        source_id: str,
        target_id: str,
        connection_type: ConnectionType,
        initial_strength: float = 0.5
    ) -> bool:
        """
        Add a connection between two agents.
        
        Args:
            source_id: Source agent ID
            target_id: Target agent ID  
            connection_type: Type of connection
            initial_strength: Initial connection strength
            
        Returns:
            True if connection was added successfully
        """
        if source_id == target_id:
            return False
        
        if source_id not in self.agents or target_id not in self.agents:
            return False
        
        # Check connection limits
        if len(self.agent_connections[source_id]) >= self.max_connections_per_agent:
            return False
        
        connection_key = (source_id, target_id)
        
        if connection_key not in self.connections:
            connection = Connection(
                source_id, target_id, connection_type,
                initial_strength, self.adaptation_rate
            )
            
            self.connections[connection_key] = connection
            self.agent_connections[source_id].add(target_id)
            self.agent_connections[target_id].add(source_id)
            
            self.logger.debug(f"Added connection: {source_id} -> {target_id}")
            return True
        
        return False
    
    def get_connections_for_agent(
        self,
        agent_id: str,
        connection_type: Optional[ConnectionType] = None,
        min_strength: float = 0.0
    ) -> List[Connection]:
        """
        Get connections for a specific agent.
        
        Args:
            agent_id: Agent ID to get connections for
            connection_type: Filter by connection type (optional)
            min_strength: Minimum connection strength
            
        Returns:
            List of matching connections
        """
        connections = []
        
        for (source_id, target_id), connection in self.connections.items():
            if source_id == agent_id or target_id == agent_id:
                if connection.strength >= min_strength:
                    if connection_type is None or connection.connection_type == connection_type:
                        if connection.is_active:
                            connections.append(connection)
        
        return connections
    
    def get_best_collaborators(
        self,
        agent_id: str,
        task_domain: Optional[str] = None,
        limit: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get the best collaborators for an agent.
        
        Args:
            agent_id: Agent ID to find collaborators for
            task_domain: Specific domain to consider
            limit: Maximum number of collaborators to return
            
        Returns:
            List of (collaborator_id, strength) tuples
        """
        connections = self.get_connections_for_agent(agent_id)
        collaborators = []
        
        for connection in connections:
            # Determine the collaborator ID
            collaborator_id = (
                connection.target_agent_id if connection.source_agent_id == agent_id
                else connection.source_agent_id
            )
            
            # Get strength for domain or general strength
            if task_domain:
                strength = connection.get_strength_for_domain(task_domain)
            else:
                strength = connection.strength
            
            collaborators.append((collaborator_id, strength))
        
        # Sort by strength and return top collaborators
        collaborators.sort(key=lambda x: x[1], reverse=True)
        return collaborators[:limit]
    
    def record_interaction(
        self,
        source_id: str,
        target_id: str,
        success: bool,
        benefit: float,
        response_time: float = 1.0,
        domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an interaction between two agents.
        
        Args:
            source_id: Source agent ID
            target_id: Target agent ID
            success: Whether interaction was successful
            benefit: Benefit gained (0.0 to 1.0)
            response_time: Time taken for interaction
            domain: Domain of interaction
            metadata: Additional metadata
        """
        connection_key = (source_id, target_id)
        reverse_key = (target_id, source_id)
        
        # Record for direct connection
        if connection_key in self.connections:
            self.connections[connection_key].record_interaction(
                success, benefit, response_time, domain, metadata
            )
        
        # Record for reverse connection if it exists
        if reverse_key in self.connections:
            self.connections[reverse_key].record_interaction(
                success, benefit, response_time, domain, metadata
            )
        
        # If no connection exists, create one if interaction was successful
        if (connection_key not in self.connections and 
            reverse_key not in self.connections and success):
            self._add_connection(
                source_id, target_id,
                ConnectionType.PEER_TO_PEER,
                initial_strength=0.3
            )
            if connection_key in self.connections:
                self.connections[connection_key].record_interaction(
                    success, benefit, response_time, domain, metadata
                )
    
    def update_connections(self, system_state: Dict[str, Any]) -> None:
        """
        Update network connections based on system state.
        
        Args:
            system_state: Current system state and performance metrics
        """
        self.iteration_count += 1
        
        # Prune weak connections periodically
        if self.iteration_count % self.pruning_interval == 0:
            self._prune_weak_connections()
        
        # Evolve network structure
        self._evolve_network_structure(system_state)
        
        # Record network state
        self._record_network_state(system_state)
    
    def _prune_weak_connections(self) -> None:
        """Remove weak or inactive connections."""
        connections_to_remove = []
        
        for key, connection in self.connections.items():
            if connection.should_prune():
                connections_to_remove.append(key)
        
        for key in connections_to_remove:
            connection = self.connections[key]
            source_id, target_id = key
            
            # Remove from tracking
            self.agent_connections[source_id].discard(target_id)
            self.agent_connections[target_id].discard(source_id)
            
            # Remove connection
            del self.connections[key]
            
            self.logger.debug(f"Pruned connection: {source_id} -> {target_id}")
    
    def _evolve_network_structure(self, system_state: Dict[str, Any]) -> None:
        """Evolve the network structure based on performance."""
        # Identify high-performing agent pairs
        high_performing_pairs = self._identify_successful_patterns()
        
        # Create new connections based on successful patterns
        self._create_new_connections(high_performing_pairs)
        
        # Adjust connection types based on performance
        self._adjust_connection_types()
    
    def _identify_successful_patterns(self) -> List[Tuple[str, str, float]]:
        """Identify patterns of successful collaborations."""
        successful_pairs = []
        
        for connection in self.connections.values():
            if (connection.metrics.interaction_count > 5 and
                connection.metrics.success_rate > 0.7):
                
                # Look for similar agent types that might benefit from connection
                source_agent = self.agents[connection.source_agent_id]
                target_agent = self.agents[connection.target_agent_id]
                
                successful_pairs.append((
                    source_agent.agent_type,
                    target_agent.agent_type,
                    connection.metrics.average_benefit
                ))
        
        return successful_pairs
    
    def _create_new_connections(
        self,
        successful_patterns: List[Tuple[str, str, float]]
    ) -> None:
        """Create new connections based on successful patterns."""
        if not successful_patterns:
            return
        
        # Group patterns by agent type pairs
        pattern_scores = defaultdict(list)
        for source_type, target_type, benefit in successful_patterns:
            pattern_scores[(source_type, target_type)].append(benefit)
        
        # Calculate average benefits for each pattern
        pattern_benefits = {
            pattern: np.mean(benefits)
            for pattern, benefits in pattern_scores.items()
        }
        
        # Create new connections for high-benefit patterns
        for (source_type, target_type), avg_benefit in pattern_benefits.items():
            if avg_benefit > 0.6:  # High benefit threshold
                self._create_connections_for_pattern(source_type, target_type, avg_benefit)
    
    def _create_connections_for_pattern(
        self,
        source_type: str,
        target_type: str,
        expected_benefit: float
    ) -> None:
        """Create connections between agents of specific types."""
        source_agents = [
            aid for aid, agent in self.agents.items()
            if agent.agent_type == source_type
        ]
        target_agents = [
            aid for aid, agent in self.agents.items()
            if agent.agent_type == target_type
        ]
        
        # Limit new connections to avoid over-connection
        max_new_connections = 2
        connections_created = 0
        
        for source_id in source_agents:
            if connections_created >= max_new_connections:
                break
                
            for target_id in target_agents:
                if connections_created >= max_new_connections:
                    break
                
                # Check if connection already exists
                if ((source_id, target_id) not in self.connections and
                    (target_id, source_id) not in self.connections):
                    
                    # Check connection limits
                    if (len(self.agent_connections[source_id]) < self.max_connections_per_agent and
                        len(self.agent_connections[target_id]) < self.max_connections_per_agent):
                        
                        initial_strength = min(0.8, 0.3 + expected_benefit * 0.3)
                        success = self._add_connection(
                            source_id, target_id,
                            ConnectionType.COLLABORATIVE,
                            initial_strength
                        )
                        
                        if success:
                            connections_created += 1
                            self.logger.debug(
                                f"Created new connection based on pattern: "
                                f"{source_type} -> {target_type}"
                            )
    
    def _adjust_connection_types(self) -> None:
        """Adjust connection types based on performance patterns."""
        for connection in self.connections.values():
            if connection.metrics.interaction_count > 10:
                success_rate = connection.metrics.success_rate
                avg_benefit = connection.metrics.average_benefit
                
                # Promote successful peer connections to collaborative
                if (connection.connection_type == ConnectionType.PEER_TO_PEER and
                    success_rate > 0.8 and avg_benefit > 0.6):
                    connection.connection_type = ConnectionType.COLLABORATIVE
                    self.logger.debug(
                        f"Promoted connection to collaborative: "
                        f"{connection.source_agent_id} -> {connection.target_agent_id}"
                    )
                
                # Demote poor collaborative connections to competitive
                elif (connection.connection_type == ConnectionType.COLLABORATIVE and
                      success_rate < 0.4):
                    connection.connection_type = ConnectionType.COMPETITIVE
                    self.logger.debug(
                        f"Demoted connection to competitive: "
                        f"{connection.source_agent_id} -> {connection.target_agent_id}"
                    )
    
    def _record_network_state(self, system_state: Dict[str, Any]) -> None:
        """Record the current state of the network."""
        active_connections = sum(1 for c in self.connections.values() if c.is_active)
        avg_strength = np.mean([c.strength for c in self.connections.values()])
        
        network_state = {
            'iteration': self.iteration_count,
            'total_connections': len(self.connections),
            'active_connections': active_connections,
            'average_strength': avg_strength,
            'agents_count': len(self.agents),
            'timestamp': time.time()
        }
        
        # Add system performance if available
        if 'current_score' in system_state:
            network_state['system_performance'] = system_state['current_score']
        
        self.evolution_history.append(network_state)
        
        # Keep limited history
        if len(self.evolution_history) > 1000:
            self.evolution_history = self.evolution_history[-1000:]
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        if not self.connections:
            return {'message': 'No connections in network'}
        
        active_connections = [c for c in self.connections.values() if c.is_active]
        
        # Connection type distribution
        type_counts = defaultdict(int)
        for connection in active_connections:
            type_counts[connection.connection_type.value] += 1
        
        # Strength statistics
        strengths = [c.strength for c in active_connections]
        
        # Agent connectivity
        agent_connectivity = {
            agent_id: len(self.agent_connections[agent_id])
            for agent_id in self.agents.keys()
        }
        
        return {
            'total_connections': len(self.connections),
            'active_connections': len(active_connections),
            'connection_types': dict(type_counts),
            'strength_stats': {
                'mean': np.mean(strengths) if strengths else 0,
                'std': np.std(strengths) if strengths else 0,
                'min': np.min(strengths) if strengths else 0,
                'max': np.max(strengths) if strengths else 0
            },
            'agent_connectivity': agent_connectivity,
            'network_density': len(active_connections) / (len(self.agents) * (len(self.agents) - 1)),
            'evolution_iterations': self.iteration_count
        }
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current network state."""
        stats = self.get_network_statistics()
        
        return {
            'active_connections': stats['active_connections'],
            'average_strength': stats['strength_stats']['mean'],
            'network_density': stats['network_density'],
            'most_connected_agent': max(
                stats['agent_connectivity'].items(),
                key=lambda x: x[1]
            )[0] if stats['agent_connectivity'] else None
        }
    
    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get the network evolution history."""
        return self.evolution_history.copy()
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for network visualization."""
        nodes = []
        edges = []
        
        # Prepare node data
        for agent_id, agent in self.agents.items():
            nodes.append({
                'id': agent_id,
                'type': agent.agent_type,
                'connections': len(self.agent_connections[agent_id])
            })
        
        # Prepare edge data
        for connection in self.connections.values():
            if connection.is_active:
                edges.append({
                    'source': connection.source_agent_id,
                    'target': connection.target_agent_id,
                    'strength': connection.strength,
                    'type': connection.connection_type.value,
                    'interactions': connection.metrics.interaction_count,
                    'success_rate': connection.metrics.success_rate
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'statistics': self.get_network_statistics()
        }
