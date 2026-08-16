import React, { useEffect, useState } from 'react';
import { Network, RefreshCw, Info, Search, Layers } from 'lucide-react';

interface NodeData {
  id: string;
  label: string;
  title: string;
  x?: number;
  y?: number;
}

interface LinkData {
  source: string;
  target: string;
  type: string;
}

const LABEL_COLORS: Record<string, string> = {
  Document: '#38bdf8', // Blue
  Act: '#f59e0b',      // Gold
  Section: '#10b981',  // Green
  Case: '#ec4899',     // Pink
  Article: '#8b5cf6',  // Purple
  Court: '#6366f1',    // Indigo
  Judge: '#14b8a6',    // Teal
  Chunk: '#64748b',    // Slate
  LegalPrinciple: '#eab308', // Yellow
  Source: '#a855f7',   // Purple-pink
};

export default function GraphVisualizer() {
  const [stats, setStats] = useState<Record<string, number>>({});
  const [nodes, setNodes] = useState<NodeData[]>([]);
  const [links, setLinks] = useState<LinkData[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState<NodeData | null>(null);
  const [neighborhood, setNeighborhood] = useState<any[] | null>(null);
  const [searchEntity, setSearchEntity] = useState('');

  const fetchGraphData = async () => {
    setLoading(true);
    try {
      // Fetch stats
      const statsRes = await fetch('/graph/stats');
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        setStats(statsData.node_counts || {});
      }

      // Fetch graph sample
      const sampleRes = await fetch('/graph/sample?limit=50');
      if (sampleRes.ok) {
        const sampleData = await sampleRes.json();
        const rawNodes: NodeData[] = sampleData.nodes || [];
        const rawLinks: LinkData[] = sampleData.links || [];

        // Simple radial layout placement algorithm for SVG visualization
        const width = 800;
        const height = 450;
        const cx = width / 2;
        const cy = height / 2;
        const radius = Math.min(width, height) * 0.38;

        const positionedNodes = rawNodes.map((node, i) => {
          const angle = (i / Math.max(rawNodes.length, 1)) * 2 * Math.PI;
          // Add small jitter for readability
          const r = radius * (0.6 + 0.4 * (i % 2));
          return {
            ...node,
            x: cx + r * Math.cos(angle),
            y: cy + r * Math.sin(angle),
          };
        });

        setNodes(positionedNodes);
        setLinks(rawLinks);
      }
    } catch (err) {
      console.error('Failed to fetch graph data:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchNeighborhood = async (entityId: string) => {
    try {
      const res = await fetch(`/graph/neighborhood/${encodeURIComponent(entityId)}`);
      if (res.ok) {
        const data = await res.json();
        setNeighborhood(data.neighbors || []);
      }
    } catch (err) {
      console.error('Failed to fetch neighborhood:', err);
    }
  };

  useEffect(() => {
    fetchGraphData();
  }, []);

  const handleNodeClick = (node: NodeData) => {
    setSelectedNode(node);
    fetchNeighborhood(node.id);
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchEntity.trim()) return;
    fetchNeighborhood(searchEntity.trim());
  };

  return (
    <div className="visualizer-container">
      {/* Node Stats Header */}
      <div className="card">
        <div className="card-title" style={{ justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Layers size={22} />
            Neo4j Knowledge Graph Statistics
          </div>
          <button className="search-button" style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }} onClick={fetchGraphData} disabled={loading}>
            <RefreshCw size={14} className={loading ? 'spinner' : ''} style={{ marginRight: '0.3rem' }} />
            Refresh
          </button>
        </div>

        <div className="stats-badges-container">
          {Object.entries(stats).length > 0 ? (
            Object.entries(stats).map(([label, count]) => (
              <div key={label} className="stat-badge" style={{ borderLeftColor: LABEL_COLORS[label] || '#38bdf8' }}>
                <span className="stat-badge-label">{label}</span>
                <span className="stat-badge-count">{count}</span>
              </div>
            ))
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>Connecting to Neo4j instance...</span>
          )}
        </div>
      </div>

      {/* Main Graph Grid */}
      <div className="results-grid">
        {/* Left Column: Interactive SVG Graph Canvas */}
        <div className="card" style={{ overflow: 'hidden' }}>
          <div className="card-title" style={{ justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Network size={20} />
              Interactive Entity & Relation Graph
            </div>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Showing {nodes.length} nodes & {links.length} connections
            </span>
          </div>

          <div className="svg-canvas-wrapper">
            <svg viewBox="0 0 800 450" className="graph-svg">
              <defs>
                <marker
                  id="arrow"
                  viewBox="0 0 10 10"
                  refX="18"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(148, 163, 184, 0.5)" />
                </marker>
              </defs>

              {/* Render Links */}
              {links.map((link, idx) => {
                const srcNode = nodes.find((n) => n.id === link.source);
                const tgtNode = nodes.find((n) => n.id === link.target);
                if (!srcNode?.x || !srcNode?.y || !tgtNode?.x || !tgtNode?.y) return null;

                return (
                  <g key={idx}>
                    <line
                      x1={srcNode.x}
                      y1={srcNode.y}
                      x2={tgtNode.x}
                      y2={tgtNode.y}
                      stroke="rgba(148, 163, 184, 0.3)"
                      strokeWidth="1.5"
                      markerEnd="url(#arrow)"
                    />
                    <text
                      x={(srcNode.x + tgtNode.x) / 2}
                      y={(srcNode.y + tgtNode.y) / 2 - 4}
                      fill="var(--text-muted)"
                      fontSize="9"
                      textAnchor="middle"
                    >
                      {link.type}
                    </text>
                  </g>
                );
              })}

              {/* Render Nodes */}
              {nodes.map((node) => {
                const color = LABEL_COLORS[node.label] || '#38bdf8';
                const isSelected = selectedNode?.id === node.id;

                return (
                  <g
                    key={node.id}
                    className="graph-node-group"
                    onClick={() => handleNodeClick(node)}
                    style={{ cursor: 'pointer' }}
                  >
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={isSelected ? 14 : 10}
                      fill={color}
                      stroke={isSelected ? '#fff' : 'rgba(15, 23, 42, 0.8)'}
                      strokeWidth={isSelected ? 3 : 1.5}
                    />
                    <text
                      x={node.x}
                      y={(node.y || 0) + 20}
                      fill="var(--text-main)"
                      fontSize="10"
                      fontWeight={isSelected ? 'bold' : 'normal'}
                      textAnchor="middle"
                    >
                      {node.title.length > 18 ? node.title.substring(0, 16) + '…' : node.title}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Right Column: Node Inspector */}
        <div className="card">
          <div className="card-title">
            <Info size={20} />
            Node & Neighborhood Inspector
          </div>

          <form onSubmit={handleSearchSubmit} style={{ display: 'flex', gap: '0.4rem', marginBottom: '1rem' }}>
            <input
              type="text"
              className="search-input"
              style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}
              placeholder="Inspect node ID (e.g. act:ipc_1860)"
              value={searchEntity}
              onChange={(e) => setSearchEntity(e.target.value)}
            />
            <button type="submit" className="search-button" style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}>
              <Search size={14} />
            </button>
          </form>

          {selectedNode ? (
            <div className="inspector-content">
              <div className="inspector-header">
                <span className="label-tag" style={{ backgroundColor: LABEL_COLORS[selectedNode.label] || '#38bdf8' }}>
                  {selectedNode.label}
                </span>
                <span className="inspector-id">{selectedNode.id}</span>
              </div>
              <h4 style={{ margin: '0.5rem 0', color: 'var(--text-main)' }}>{selectedNode.title}</h4>

              {neighborhood && neighborhood.length > 0 && (
                <div style={{ marginTop: '1rem' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-blue)', marginBottom: '0.5rem' }}>
                    Graph Connections ({neighborhood.length})
                  </div>
                  <div className="neighborhood-list">
                    {neighborhood.map((rel, idx) => (
                      <div key={idx} className="neighborhood-item">
                        <span className="rel-badge">{rel.relation || 'RELATED'}</span>
                        <span className="target-name">{rel.target_id || rel.source_id}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', fontStyle: 'italic' }}>
              Click any node on the graph canvas or type an entity ID above to inspect its properties and relationships.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
