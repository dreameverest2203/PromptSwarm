"""
Network visualization utilities for the MPEN system.
"""

from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class NetworkVisualizer:
    """
    Visualizes the adaptive network structure and evolution.
    
    Provides both static (matplotlib) and interactive (plotly) visualizations
    of agent networks, connection strengths, and evolution over time.
    """
    
    def __init__(self):
        """Initialize the network visualizer."""
        self.color_map = {
            'GeneratorAgent': '#FF6B6B',
            'CriticAgent': '#4ECDC4', 
            'ValidatorAgent': '#45B7D1',
            'MetaAgent': '#96CEB4'
        }
        
        self.connection_color_map = {
            'collaborative': '#2ECC71',
            'competitive': '#E74C3C',
            'hierarchical': '#9B59B6',
            'peer_to_peer': '#3498DB',
            'mentoring': '#F39C12',
            'validation': '#1ABC9C'
        }
    
    def plot_network_structure(
        self,
        network_data: Dict[str, Any],
        save_path: Optional[str] = None,
        figsize: tuple = (12, 8)
    ) -> plt.Figure:
        """
        Plot the network structure using matplotlib.
        
        Args:
            network_data: Network visualization data
            save_path: Path to save the figure (optional)
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for node in network_data['nodes']:
            G.add_node(
                node['id'],
                type=node['type'],
                connections=node['connections']
            )
        
        # Add edges
        for edge in network_data['edges']:
            G.add_edge(
                edge['source'],
                edge['target'],
                strength=edge['strength'],
                type=edge['type']
            )
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes by type
        for agent_type, color in self.color_map.items():
            nodes_of_type = [
                node for node, data in G.nodes(data=True)
                if data['type'] == agent_type
            ]
            if nodes_of_type:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=nodes_of_type,
                    node_color=color,
                    node_size=1000,
                    alpha=0.8,
                    ax=ax
                )
        
        # Draw edges by connection type and strength
        for conn_type, color in self.connection_color_map.items():
            edges_of_type = [
                (u, v) for u, v, data in G.edges(data=True)
                if data['type'] == conn_type
            ]
            
            if edges_of_type:
                # Get edge strengths for width
                edge_strengths = [
                    G[u][v]['strength'] for u, v in edges_of_type
                ]
                edge_widths = [strength * 5 for strength in edge_strengths]
                
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=edges_of_type,
                    edge_color=color,
                    width=edge_widths,
                    alpha=0.6,
                    ax=ax
                )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        # Create legend
        legend_elements = []
        
        # Agent type legend
        for agent_type, color in self.color_map.items():
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color, markersize=10,
                          label=agent_type)
            )
        
        # Connection type legend  
        for conn_type, color in self.connection_color_map.items():
            legend_elements.append(
                plt.Line2D([0], [0], color=color, linewidth=3,
                          label=f"{conn_type} connection")
            )
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        ax.set_title("MPEN Agent Network Structure", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_interactive_network(
        self,
        network_data: Dict[str, Any]
    ) -> Optional[go.Figure]:
        """
        Create an interactive network plot using plotly.
        
        Args:
            network_data: Network visualization data
            
        Returns:
            Plotly figure object or None if plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return None
        
        # Create NetworkX graph for layout
        G = nx.Graph()
        
        for node in network_data['nodes']:
            G.add_node(node['id'], **node)
        
        for edge in network_data['edges']:
            G.add_edge(edge['source'], edge['target'], **edge)
        
        # Get positions
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Prepare node traces
        node_traces = []
        
        for agent_type, color in self.color_map.items():
            nodes_of_type = [
                node for node in network_data['nodes']
                if node['type'] == agent_type
            ]
            
            if nodes_of_type:
                x_coords = [pos[node['id']][0] for node in nodes_of_type]
                y_coords = [pos[node['id']][1] for node in nodes_of_type]
                
                hover_text = [
                    f"ID: {node['id']}<br>"
                    f"Type: {node['type']}<br>"
                    f"Connections: {node['connections']}"
                    for node in nodes_of_type
                ]
                
                node_trace = go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color=color,
                        line=dict(width=2, color='black')
                    ),
                    text=[node['id'] for node in nodes_of_type],
                    textposition="middle center",
                    hovertext=hover_text,
                    hoverinfo='text',
                    name=agent_type
                )
                node_traces.append(node_trace)
        
        # Prepare edge traces
        edge_traces = []
        
        for conn_type, color in self.connection_color_map.items():
            edges_of_type = [
                edge for edge in network_data['edges']
                if edge['type'] == conn_type
            ]
            
            if edges_of_type:
                edge_x = []
                edge_y = []
                
                for edge in edges_of_type:
                    x0, y0 = pos[edge['source']]
                    x1, y1 = pos[edge['target']]
                    
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=2, color=color),
                    hoverinfo='none',
                    mode='lines',
                    name=f"{conn_type} connections"
                )
                edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(data=edge_traces + node_traces)
        
        fig.update_layout(
            title="Interactive MPEN Agent Network",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Hover over nodes for details",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="gray", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def plot_network_evolution(
        self,
        evolution_history: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        figsize: tuple = (12, 8)
    ) -> plt.Figure:
        """
        Plot network evolution over time.
        
        Args:
            evolution_history: List of network state snapshots
            save_path: Path to save the figure (optional)
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        if not evolution_history:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No evolution history available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        iterations = [state['iteration'] for state in evolution_history]
        
        # Plot 1: Connection counts over time
        total_connections = [state['total_connections'] for state in evolution_history]
        active_connections = [state['active_connections'] for state in evolution_history]
        
        ax1.plot(iterations, total_connections, label='Total Connections', marker='o')
        ax1.plot(iterations, active_connections, label='Active Connections', marker='s')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Connection Count')
        ax1.set_title('Network Size Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average connection strength
        avg_strengths = [state['average_strength'] for state in evolution_history]
        
        ax2.plot(iterations, avg_strengths, label='Average Strength', 
                color='green', marker='d')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Average Strength')
        ax2.set_title('Connection Strength Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: System performance (if available)
        if 'system_performance' in evolution_history[0]:
            performances = [
                state.get('system_performance', 0) 
                for state in evolution_history
            ]
            ax3.plot(iterations, performances, label='System Performance',
                    color='purple', marker='^')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Performance Score')
            ax3.set_title('System Performance vs Network Evolution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'System performance\ndata not available',
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Network density
        if 'agents_count' in evolution_history[0]:
            densities = []
            for state in evolution_history:
                n_agents = state['agents_count']
                max_connections = n_agents * (n_agents - 1)
                density = state['active_connections'] / max_connections if max_connections > 0 else 0
                densities.append(density)
            
            ax4.plot(iterations, densities, label='Network Density',
                    color='orange', marker='x')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Density')
            ax4.set_title('Network Density Evolution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Network density\ndata not available',
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_connection_strength_heatmap(
        self,
        network_data: Dict[str, Any],
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8)
    ) -> plt.Figure:
        """
        Plot a heatmap of connection strengths between agents.
        
        Args:
            network_data: Network visualization data
            save_path: Path to save the figure (optional)
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        # Get agent IDs
        agent_ids = [node['id'] for node in network_data['nodes']]
        n_agents = len(agent_ids)
        
        # Create strength matrix
        strength_matrix = np.zeros((n_agents, n_agents))
        
        agent_id_to_idx = {agent_id: i for i, agent_id in enumerate(agent_ids)}
        
        for edge in network_data['edges']:
            source_idx = agent_id_to_idx[edge['source']]
            target_idx = agent_id_to_idx[edge['target']]
            strength = edge['strength']
            
            strength_matrix[source_idx, target_idx] = strength
            strength_matrix[target_idx, source_idx] = strength  # Symmetric
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(strength_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(n_agents))
        ax.set_yticks(range(n_agents))
        ax.set_xticklabels(agent_ids, rotation=45, ha='right')
        ax.set_yticklabels(agent_ids)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Connection Strength', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(n_agents):
            for j in range(n_agents):
                if strength_matrix[i, j] > 0:
                    text = ax.text(j, i, f'{strength_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title("Agent Connection Strength Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dashboard(
        self,
        network_data: Dict[str, Any],
        evolution_history: List[Dict[str, Any]]
    ) -> Optional[go.Figure]:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            network_data: Current network state data
            evolution_history: Network evolution history
            
        Returns:
            Plotly dashboard figure or None if plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Network Structure", "Connection Evolution",
                "Agent Connectivity", "Performance Metrics"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # Network structure (simplified)
        agent_types = [node['type'] for node in network_data['nodes']]
        type_counts = {agent_type: agent_types.count(agent_type) 
                      for agent_type in set(agent_types)}
        
        fig.add_trace(
            go.Bar(x=list(type_counts.keys()), y=list(type_counts.values()),
                  name="Agent Types"),
            row=1, col=1
        )
        
        # Connection evolution
        if evolution_history:
            iterations = [state['iteration'] for state in evolution_history]
            active_connections = [state['active_connections'] for state in evolution_history]
            
            fig.add_trace(
                go.Scatter(x=iterations, y=active_connections, 
                          mode='lines+markers', name="Active Connections"),
                row=1, col=2
            )
        
        # Agent connectivity
        connectivity = [node['connections'] for node in network_data['nodes']]
        agent_ids = [node['id'] for node in network_data['nodes']]
        
        fig.add_trace(
            go.Bar(x=agent_ids, y=connectivity, name="Connections per Agent"),
            row=2, col=1
        )
        
        # Performance metrics
        if evolution_history and 'system_performance' in evolution_history[0]:
            performances = [state.get('system_performance', 0) for state in evolution_history]
            fig.add_trace(
                go.Scatter(x=iterations, y=performances,
                          mode='lines+markers', name="System Performance"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="MPEN Network Dashboard",
            showlegend=True,
            height=800
        )
        
        return fig
