// src/components/FullGraphView.tsx
import React, { useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { forceCollide, forceX, forceY } from 'd3-force';
import { X, Loader2, Search } from 'lucide-react';
import { FullGraphData } from '../types';
import { getFullGraph } from '../services/api';

interface FullGraphViewProps {
  onClose: () => void;
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

const REL_COLORS: Record<string, string> = {
  TREATS: '#16a34a',
  TREATED_BY: '#16a34a',
  CONTRAINDICATED_IN: '#dc2626',
  CONTRAINDICATES: '#dc2626',
  REQUIRES_ADJUSTMENT: '#d97706',
  ADJUSTMENT_REQUIRED_FOR: '#d97706',
  INTERACTS_WITH: '#7c3aed',
};
const FALLBACK_REL_COLORS = ['#0891b2', '#db2777', '#65a30d', '#4f46e5', '#ea580c'];

const getRelColor = (type: string) => {
  if (REL_COLORS[type]) return REL_COLORS[type];
  let hash = 0;
  for (let i = 0; i < type.length; i++) hash = (hash * 31 + type.charCodeAt(i)) >>> 0;
  return FALLBACK_REL_COLORS[hash % FALLBACK_REL_COLORS.length];
};

// After the sim starts, d3-force mutates link.source/target from a plain id
// string into a reference to the actual node object - handle either shape.
const endpointId = (endpoint: any): string => (endpoint && typeof endpoint === 'object' ? endpoint.id : endpoint);

export const FullGraphView: React.FC<FullGraphViewProps> = ({ onClose }) => {
  const graphRef = React.useRef<any>();
  const [data, setData] = useState<FullGraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [highlightedIds, setHighlightedIds] = useState<Set<string>>(new Set());
  const highlightTimeoutRef = React.useRef<ReturnType<typeof setTimeout>>();

  const [relNode, setRelNode] = useState('');
  const [relType, setRelType] = useState('');
  const [relResults, setRelResults] = useState<{ id: string; name: string; type: string }[] | null>(null);
  const [relMessage, setRelMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getFullGraph()
      .then(d => {
        if (!cancelled) setData(d);
      })
      .catch(() => {
        if (!cancelled) setError('Could not load the full knowledge graph.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  const graphData = React.useMemo(
    () => ({
      nodes: data?.nodes.map(n => ({ id: n.id, name: n.name, type: n.type })) ?? [],
      links: data?.links.map(l => ({ source: l.source, target: l.target, label: l.relationship })) ?? [],
    }),
    [data]
  );

  const presentTypes = React.useMemo(
    () => Array.from(new Set(data?.nodes.map(n => n.type) ?? [])).sort(),
    [data]
  );

  const relationshipTypes = React.useMemo(
    () => Array.from(new Set(data?.links.map(l => l.relationship) ?? [])).sort(),
    [data]
  );

  useEffect(() => {
    const fg = graphRef.current;
    if (!fg) return;
    fg.d3Force('charge')?.strength(-60);
    fg.d3Force('link')?.distance(60);
    fg.d3Force('collide', forceCollide(16));
    // The graph is mostly small disconnected fragments plus one dense hub;
    // with nothing pulling separate components together, repulsion alone lets
    // them drift arbitrarily far apart, so zoomToFit wastes most of the view
    // on empty space. A gentle pull toward center keeps everything bounded.
    fg.d3Force('x', forceX(0).strength(0.03));
    fg.d3Force('y', forceY(0).strength(0.03));
  }, [graphData]);

  const searchMatches = React.useMemo(() => {
    const term = searchTerm.trim().toLowerCase();
    if (!term) return [];
    return graphData.nodes.filter(n => n.name.toLowerCase().includes(term)).slice(0, 8);
  }, [searchTerm, graphData]);

  const clearHighlightLater = () => {
    if (highlightTimeoutRef.current) clearTimeout(highlightTimeoutRef.current);
    highlightTimeoutRef.current = setTimeout(() => setHighlightedIds(new Set()), 5000);
  };

  const focusNode = (nodeId: string) => {
    const fg = graphRef.current;
    if (!fg) return;

    // `graphData()` isn't exposed on the ref, but d3-force mutates our node
    // objects in place, so the ones already in `graphData.nodes` carry live x/y.
    const liveNode = graphData.nodes.find(n => n.id === nodeId) as any;
    if (!liveNode || liveNode.x === undefined) return;

    fg.centerAt(liveNode.x, liveNode.y, 600);
    fg.zoom(5, 600);

    setHighlightedIds(new Set([nodeId]));
    setSearchTerm('');
    clearHighlightLater();
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchMatches.length > 0) focusNode(searchMatches[0].id);
  };

  const handleRelSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setRelResults(null);
    setRelMessage(null);

    const nodeName = relNode.trim();
    const type = relType.trim();
    if (!nodeName || !type) return;

    const matchedNode = graphData.nodes.find(n => n.name.toLowerCase() === nodeName.toLowerCase());
    if (!matchedNode) {
      setRelMessage(`No node named "${nodeName}" found.`);
      return;
    }

    const neighborIds = new Set<string>();
    graphData.links.forEach(l => {
      if (l.label !== type) return;
      const source = endpointId(l.source);
      const target = endpointId(l.target);
      if (source === matchedNode.id) neighborIds.add(target);
      else if (target === matchedNode.id) neighborIds.add(source);
    });

    const results = graphData.nodes.filter(n => neighborIds.has(n.id));
    setRelResults(results);

    if (results.length === 0) {
      setRelMessage(`No nodes found with a "${type.replace(/_/g, ' ')}" relationship to "${matchedNode.name}".`);
      return;
    }

    const allIds = new Set([matchedNode.id, ...results.map(r => r.id)]);
    setHighlightedIds(allIds);
    clearHighlightLater();
    graphRef.current?.zoomToFit(600, 80, (n: any) => allIds.has(n.id));
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full h-full max-w-[95vw] max-h-[95vh] flex flex-col overflow-hidden">
        <div className="flex items-center justify-between gap-6 px-8 py-6 border-b border-gray-200">
          <div>
            <h2 className="text-3xl font-bold text-gray-800">Full Knowledge Graph</h2>
            <p className="text-base text-gray-500">Drag to rearrange, scroll to zoom, hover a node for its name.</p>
          </div>

          <div className="relative flex-1 max-w-sm">
            <form onSubmit={handleSearchSubmit}>
              <Search className="w-5 h-5 text-gray-400 absolute left-4 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                placeholder="Find a node..."
                className="w-full pl-11 pr-4 py-3 text-base border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
              />
            </form>

            {searchMatches.length > 0 && (
              <ul className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-20 max-h-56 overflow-y-auto">
                {searchMatches.map(n => (
                  <li key={n.id}>
                    <button
                      onClick={() => focusNode(n.id)}
                      className="w-full text-left px-4 py-2.5 text-base hover:bg-gray-100 flex items-center gap-2"
                    >
                      <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: getNodeColor(n.type) }} />
                      {n.name}
                    </button>
                  </li>
                ))}
              </ul>
            )}
            {searchTerm.trim() && searchMatches.length === 0 && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-20 px-4 py-2.5 text-base text-gray-400">
                No matches
              </div>
            )}
          </div>

          <button
            onClick={onClose}
            className="p-2.5 text-gray-500 hover:bg-gray-100 rounded-lg transition-all"
            aria-label="Close"
          >
            <X className="w-7 h-7" />
          </button>
        </div>

        {/* Relationship explorer: find every node connected to a given node via a given relationship */}
        <div className="px-8 py-4 border-b border-gray-200 bg-gray-50">
          <form onSubmit={handleRelSearchSubmit} className="flex flex-wrap items-center gap-3">
            <span className="text-base text-gray-500">Find nodes where</span>
            <input
              type="text"
              list="full-graph-node-names"
              value={relNode}
              onChange={e => setRelNode(e.target.value)}
              placeholder="e.g. Metformin"
              className="px-3 py-2 text-base border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 w-48"
            />
            <datalist id="full-graph-node-names">
              {graphData.nodes.map(n => (
                <option key={n.id} value={n.name} />
              ))}
            </datalist>
            <select
              value={relType}
              onChange={e => setRelType(e.target.value)}
              className="px-3 py-2 text-base border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
            >
              <option value="">select relationship...</option>
              {relationshipTypes.map(type => (
                <option key={type} value={type}>{type.replace(/_/g, ' ')}</option>
              ))}
            </select>
            <button
              type="submit"
              disabled={!relNode.trim() || !relType}
              className="px-4 py-2 text-base bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 transition-all"
            >
              Find
            </button>

            {relMessage && <span className="text-base text-gray-500">{relMessage}</span>}
            {relResults && relResults.length > 0 && (
              <span className="text-base text-gray-700">
                Found {relResults.length}: {relResults.map(r => r.name).join(', ')}
              </span>
            )}
          </form>
        </div>

        <div className="relative flex-1">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center gap-2 text-gray-500 z-10 bg-white/70">
              <Loader2 className="w-5 h-5 animate-spin" />
              Loading the full graph...
            </div>
          )}
          {error && !loading && (
            <div className="absolute inset-0 flex items-center justify-center text-red-600 text-sm z-10 bg-white/70 px-6 text-center">
              {error}
            </div>
          )}

          <ForceGraph2D
            ref={graphRef}
            graphData={graphData}
            nodeLabel="name"
            nodeCanvasObject={(node: any, ctx, globalScale) => {
              const radius = 5;
              const isFound = highlightedIds.has(node.id);

              if (isFound) {
                ctx.beginPath();
                ctx.arc(node.x, node.y, radius + 5, 0, 2 * Math.PI);
                ctx.strokeStyle = '#facc15';
                ctx.lineWidth = 2.5 / globalScale;
                ctx.stroke();
              }

              ctx.beginPath();
              ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
              ctx.fillStyle = getNodeColor(node.type);
              ctx.fill();

              const fontSize = (isFound ? 10 : 8) / globalScale;
              ctx.font = `${isFound ? 'bold ' : ''}${fontSize}px Sans-Serif`;
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillStyle = isFound ? '#b45309' : '#111827';
              ctx.fillText(node.name, node.x, node.y + radius + fontSize + 1);
            }}
            linkLabel="label"
            linkDirectionalArrowLength={3}
            linkDirectionalArrowRelPos={1}
            linkCurvature={0.2}
            linkColor={(link: any) => getRelColor(link.label)}
            linkWidth={1.2}
            linkCanvasObjectMode={() => 'after'}
            linkCanvasObject={(link: any, ctx, globalScale) => {
              const start = link.source;
              const end = link.target;
              if (typeof start !== 'object' || typeof end !== 'object' || start.x === undefined || end.x === undefined) return;

              const dx = end.x - start.x;
              const dy = end.y - start.y;
              const midX = (start.x + end.x) / 2 - dy * 0.1;
              const midY = (start.y + end.y) / 2 + dx * 0.1;

              const label = link.label.replace(/_/g, ' ');
              const fontSize = 6 / globalScale;
              ctx.font = `${fontSize}px Sans-Serif`;
              const padding = 1.2 / globalScale;
              const textWidth = ctx.measureText(label).width;

              ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
              ctx.fillRect(midX - textWidth / 2 - padding, midY - fontSize / 2 - padding, textWidth + padding * 2, fontSize + padding * 2);

              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillStyle = getRelColor(link.label);
              ctx.fillText(label, midX, midY);
            }}
            backgroundColor="#ffffff"
            d3VelocityDecay={0.3}
            cooldownTime={4000}
            onEngineStop={() => graphRef.current?.zoomToFit(400, 40)}
          />
        </div>

        <div className="px-6 py-3 border-t border-gray-200 flex flex-wrap gap-x-6 gap-y-2">
          {presentTypes.map(type => (
            <div key={type} className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: getNodeColor(type) }} />
              <span className="text-sm text-gray-600">{type}</span>
            </div>
          ))}
          {relationshipTypes.map(type => (
            <div key={type} className="flex items-center gap-2">
              <div className="w-4 h-0.5 rounded" style={{ backgroundColor: getRelColor(type) }} />
              <span className="text-sm text-gray-600">{type.replace(/_/g, ' ')}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
