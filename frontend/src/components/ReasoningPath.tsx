// src/components/ReasoningPath.tsx
import React from 'react';
import { Path } from '../types';
import { ArrowRight, Star } from 'lucide-react';

interface ReasoningPathProps {
  paths: Path[];
}

export const ReasoningPath: React.FC<ReasoningPathProps> = ({ paths }) => {
  const getNodeColor = (type: string) => {
    const colors: Record<string, string> = {
      Drug: 'bg-blue-100 text-blue-800 border-blue-300',
      Disease: 'bg-orange-100 text-orange-800 border-orange-300',
      SideEffect: 'bg-red-100 text-red-800 border-red-300',
      Target: 'bg-green-100 text-green-800 border-green-300',
      Pathway: 'bg-purple-100 text-purple-800 border-purple-300',
    };
    return colors[type] || 'bg-gray-100 text-gray-800 border-gray-300';
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl p-6 mb-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Reasoning Paths</h2>
      
      <div className="space-y-6">
        {paths.slice(0, 3).map((path, pathIdx) => (
          <div key={pathIdx} className="border border-gray-200 rounded-xl p-4">
            {/* Path header */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Star className="w-5 h-5 text-yellow-500 fill-yellow-500" />
                <span className="font-semibold text-gray-700">
                  Path {pathIdx + 1}
                </span>
              </div>
              <div className="text-sm">
                <span className="text-gray-600">Score: </span>
                <span className="font-bold text-primary">{path.score.toFixed(1)}</span>
              </div>
            </div>

            {/* Path visualization */}
            <div className="flex items-center flex-wrap gap-2">
              {path.nodes.map((node, nodeIdx) => (
                <React.Fragment key={nodeIdx}>
                  {/* Node */}
                  <div className={`px-4 py-2 rounded-lg border-2 ${getNodeColor(path.node_types[nodeIdx])}`}>
                    <div className="font-semibold">{node}</div>
                    <div className="text-xs opacity-75">{path.node_types[nodeIdx]}</div>
                  </div>
                  
                  {/* Arrow with relationship */}
                  {nodeIdx < path.relationships.length && (
                    <div className="flex flex-col items-center">
                      <ArrowRight className="w-6 h-6 text-gray-400" />
                      <span className="text-xs text-gray-500 mt-1 max-w-[120px] text-center">
                        {path.relationships[nodeIdx].replace(/_/g, ' ')}
                      </span>
                    </div>
                  )}
                </React.Fragment>
              ))}
            </div>

            {/* Evidence */}
            {path.evidence && path.evidence.length > 0 && (
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <div className="text-sm font-semibold text-gray-700 mb-2">Evidence:</div>
                <ul className="text-sm text-gray-600 space-y-1">
                  {path.evidence.map((ev, evIdx) => (
                    <li key={evIdx} className="flex items-start">
                      <span className="mr-2">•</span>
                      <span>{ev}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};