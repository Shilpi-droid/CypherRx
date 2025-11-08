// src/components/GraphVisualization.tsx
import React, { useEffect, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Path } from '../types';

interface GraphVisualizationProps {
  paths: Path[];
}

export const GraphVisualization: React.FC<GraphVisualizationProps> = ({ paths }) => {
  const graphRef = useRef<any>();

  // Convert paths to graph data
  const graphData = React.useMemo(() => {
    const nodes = new Map();
    const links: any[] = [];

    paths.forEach((path, pathIdx) => {
      path.nodes.forEach((node, nodeIdx) => {
        if (!nodes.has(node)) {
          nodes.set(node, {
            id: node,
            name: node,
            type: path.node_types[nodeIdx],
            val: 1
          });
        } else {
          // Increase node size if it appears in multiple paths
          const existing = nodes.get(node);
          existing.val += 0.5;
        }

        // Add links
        if (nodeIdx < path.nodes.length - 1) {
          links.push({
            source: node,
            target: path.nodes[nodeIdx + 1],
            label: path.relationships[nodeIdx],
            pathIdx
          });
        }
      });
    });

    return {
      nodes: Array.from(nodes.values()),
      links
    };
  }, [paths]);

  const getNodeColor = (type: string) => {
    const colors: Record<string, string> = {
      Drug: '#3b82f6',
      Disease: '#f97316',
      SideEffect: '#ef4444',
      Target: '#10b981',
      Pathway: '#8b5cf6',
    };
    return colors[type] || '#6b7280';
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl p-6 mb-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Knowledge Graph Visualization</h2>
      
      <div className="border border-gray-200 rounded-xl overflow-hidden" style={{ height: '500px' }}>
        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          nodeLabel="name"
          nodeAutoColorBy="type"
          nodeCanvasObject={(node: any, ctx, globalScale) => {
            const label = node.name;
            const fontSize = 12 / globalScale;
            ctx.font = `${fontSize}px Sans-Serif`;
            
            // Draw node
            ctx.fillStyle = getNodeColor(node.type);
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.val * 5, 0, 2 * Math.PI);
            ctx.fill();
            
            // Draw label
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#000';
            ctx.fillText(label, node.x, node.y + node.val * 5 + fontSize + 2);
          }}
          linkLabel="label"
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={1}
          linkCurvature={0.25}
          linkColor={() => '#94a3b8'}
          linkWidth={2}
          backgroundColor="#ffffff"
          d3VelocityDecay={0.3}
        />
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4">
        {['Drug', 'Disease', 'SideEffect', 'Target', 'Pathway'].map(type => (
          <div key={type} className="flex items-center gap-2">
            <div 
              className="w-4 h-4 rounded-full" 
              style={{ backgroundColor: getNodeColor(type) }}
            />
            <span className="text-sm text-gray-600">{type}</span>
          </div>
        ))}
      </div>
    </div>
  );
};