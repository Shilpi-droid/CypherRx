// src/components/GraphVisualization.tsx
import React, { useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { forceCollide } from 'd3-force';
import { Maximize2 } from 'lucide-react';
import { Path } from '../types';
import { getNodeNeighbors } from '../services/api';

interface GraphVisualizationProps {
  paths: Path[];
  compact?: boolean;
  onExpand?: () => void;
}

interface GraphNode {
  id: string;
  name: string;
  type: string;
}

interface GraphLink {
  source: string;
  target: string;
  label: string;
}

const getNodeColor = (type: string) => {
  const colors: Record<string, string> = {
    Drug: '#3b82f6',
    Disease: '#f97316',
    Condition: '#f97316',
    SideEffect: '#ef4444',
    Target: '#10b981',
    Pathway: '#8b5cf6',
  };
  return colors[type] || '#6b7280';
};

// Undirected pair key so we don't draw a second, opposite-facing edge for a
// relationship that's already shown (e.g. a path already displays
// Drug-Condition; a later expansion re-discovering the same edge shouldn't
// add a duplicate parallel link).
const pairKey = (a: string, b: string) => [a, b].sort().join('|||');

const buildBaseGraph = (paths: Path[]): { nodes: Map<string, GraphNode>; links: GraphLink[] } => {
  const nodes = new Map<string, GraphNode>();
  const links: GraphLink[] = [];
  const seenPairs = new Set<string>();

  paths.forEach(path => {
    path.nodes.forEach((name, idx) => {
      if (!nodes.has(name)) {
        nodes.set(name, { id: name, name, type: path.node_types[idx] });
      }
    });
    for (let i = 0; i < path.nodes.length - 1; i++) {
      const a = path.nodes[i];
      const b = path.nodes[i + 1];
      const key = pairKey(a, b);
      if (!seenPairs.has(key)) {
        seenPairs.add(key);
        links.push({ source: a, target: b, label: path.relationships[i] });
      }
    }
  });

  return { nodes, links };
};

export const GraphVisualization: React.FC<GraphVisualizationProps> = ({ paths, compact = false, onExpand }) => {
  const graphRef = React.useRef<any>();
  const [nodes, setNodes] = useState<Map<string, GraphNode>>(new Map());
  const [links, setLinks] = useState<GraphLink[]>([]);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [expandError, setExpandError] = useState<string | null>(null);
  const [hovering, setHovering] = useState(false);

  // Reset to the fresh answer's reasoning subgraph whenever a new query result comes in.
  useEffect(() => {
    const base = buildBaseGraph(paths);
    setNodes(base.nodes);
    setLinks(base.links);
    setExpandedIds(new Set());
    setExpandError(null);
  }, [paths]);

  // Default forces pack nodes too tightly once a few expansions are added -
  // circles start overlapping and edge labels collide. Push nodes further
  // apart and hard-stop them from overlapping outright.
  // In compact mode the canvas is much smaller (~280px in a narrow column),
  // so the same spacing used for the full-size panel would produce a
  // bounding box zoomToFit has to shrink so much the nodes become
  // imperceptibly small - scale the forces down to match.
  useEffect(() => {
    const fg = graphRef.current;
    if (!fg) return;
    fg.d3Force('charge')?.strength(compact ? -90 : -220);
    fg.d3Force('link')?.distance(compact ? 50 : 110);
    fg.d3Force('collide', forceCollide(compact ? 14 : 24));
  }, [nodes, links, compact]);

  const handleNodeClick = React.useCallback(
    async (node: any) => {
      if (loadingId || expandedIds.has(node.id)) return;

      setLoadingId(node.id);
      setExpandError(null);

      try {
        const data = await getNodeNeighbors(node.id, node.type);

        setNodes(prev => {
          const next = new Map(prev);
          data.nodes.forEach(n => {
            if (!next.has(n.id)) next.set(n.id, { id: n.id, name: n.name, type: n.type });
          });
          return next;
        });

        setLinks(prev => {
          const seen = new Set(prev.map(l => pairKey(l.source, l.target)));
          const additions: GraphLink[] = [];
          data.links.forEach(l => {
            const key = pairKey(l.source, l.target);
            if (!seen.has(key)) {
              seen.add(key);
              additions.push({ source: l.source, target: l.target, label: l.relationship });
            }
          });
          return [...prev, ...additions];
        });

        setExpandedIds(prev => new Set(prev).add(node.id));
      } catch {
        setExpandError(`Could not expand "${node.name}".`);
      } finally {
        setLoadingId(null);
      }
    },
    [loadingId, expandedIds]
  );

  const graphData = React.useMemo(
    () => ({
      nodes: Array.from(nodes.values()).map(n => ({
        ...n,
        expanded: expandedIds.has(n.id),
        loading: loadingId === n.id,
      })),
      links,
    }),
    [nodes, links, expandedIds, loadingId]
  );

  return (
    <div className="bg-white rounded-2xl shadow-xl p-6 mb-6">
      <div className="flex items-center justify-between mb-1 flex-wrap gap-2">
        <h2 className={compact ? 'text-lg font-bold text-gray-800' : 'text-2xl font-bold text-gray-800'}>
          Knowledge Graph{compact ? '' : ' Visualization'}
        </h2>
        <div className="flex items-center gap-2">
          {loadingId && <span className="text-xs text-gray-500">Expanding {loadingId}...</span>}
          {onExpand && (
            <button
              onClick={onExpand}
              className="p-1.5 text-gray-500 hover:bg-gray-100 rounded-lg transition-all"
              aria-label="View full graph"
              title="View full graph"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
      {!compact && (
        <p className="text-sm text-gray-500 mb-4">Click a node to expand its connections. Drag to rearrange, scroll to zoom.</p>
      )}

      <div
        className="border border-gray-200 rounded-xl overflow-hidden relative mt-2"
        style={{ height: compact ? '280px' : '500px', cursor: hovering ? 'pointer' : 'grab' }}
      >
        {expandError && (
          <div className="absolute top-2 left-1/2 -translate-x-1/2 bg-red-50 text-red-600 text-xs px-3 py-1.5 rounded-full shadow z-10">
            {expandError}
          </div>
        )}

        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          nodeLabel="name"
          onNodeClick={handleNodeClick}
          onNodeHover={(node: any) => setHovering(!!node)}
          nodeCanvasObject={(node: any, ctx, globalScale) => {
            const radius = 6;
            const color = node.loading ? '#f59e0b' : getNodeColor(node.type);

            ctx.beginPath();
            ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();

            // Nodes not yet expanded get a ring, hinting there's more to explore
            if (!node.expanded && !node.loading) {
              ctx.lineWidth = 1.5 / globalScale;
              ctx.strokeStyle = color;
              ctx.globalAlpha = 0.4;
              ctx.beginPath();
              ctx.arc(node.x, node.y, radius + 3, 0, 2 * Math.PI);
              ctx.stroke();
              ctx.globalAlpha = 1;
            }

            const fontSize = 12 / globalScale;
            ctx.font = `${fontSize}px Sans-Serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#111827';
            ctx.fillText(node.name, node.x, node.y + radius + fontSize + 2);
          }}
          linkLabel="label"
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={1}
          linkCurvature={0.2}
          linkColor={() => '#94a3b8'}
          linkWidth={1.5}
          linkCanvasObjectMode={() => 'after'}
          linkCanvasObject={(link: any, ctx, globalScale) => {
            const start = link.source;
            const end = link.target;
            if (typeof start !== 'object' || typeof end !== 'object' || start.x === undefined || end.x === undefined) return;

            // Approximate the midpoint of the rendered curve (linkCurvature=0.2):
            // offset perpendicular to the straight line between the two nodes.
            const dx = end.x - start.x;
            const dy = end.y - start.y;
            const midX = (start.x + end.x) / 2 - dy * 0.1;
            const midY = (start.y + end.y) / 2 + dx * 0.1;

            const label = link.label.replace(/_/g, ' ');
            const fontSize = 8 / globalScale;
            ctx.font = `${fontSize}px Sans-Serif`;
            const padding = 1.5 / globalScale;
            const textWidth = ctx.measureText(label).width;

            ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
            ctx.fillRect(midX - textWidth / 2 - padding, midY - fontSize / 2 - padding, textWidth + padding * 2, fontSize + padding * 2);

            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#475569';
            ctx.fillText(label, midX, midY);
          }}
          backgroundColor="#ffffff"
          d3VelocityDecay={0.3}
          cooldownTime={3500}
          onEngineStop={() => graphRef.current?.zoomToFit(400, compact ? 20 : 60)}
        />
      </div>

      {/* Legend */}
      <div className={compact ? 'mt-3 flex flex-wrap gap-3' : 'mt-4 flex flex-wrap gap-4'}>
        {['Drug', 'Condition', 'SideEffect', 'Target', 'Pathway'].map(type => (
          <div key={type} className="flex items-center gap-1.5">
            <div
              className={compact ? 'w-2.5 h-2.5 rounded-full' : 'w-4 h-4 rounded-full'}
              style={{ backgroundColor: getNodeColor(type) }}
            />
            <span className={compact ? 'text-xs text-gray-600' : 'text-sm text-gray-600'}>{type}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
