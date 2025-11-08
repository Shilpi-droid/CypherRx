// src/components/QueryHistory.tsx
import React from 'react';
import { History, Trash2, Clock } from 'lucide-react';
import { HistoryEntry } from '../types';

interface QueryHistoryProps {
  queryHistory?: HistoryEntry[];
  history?: HistoryEntry[];
  onQuerySelect?: (query: string) => void;
  onSelectQuery?: (query: string) => void;
  onClearHistory?: () => void;
  onClear?: () => void;
  isInSidebar?: boolean;
}

export const QueryHistory: React.FC<QueryHistoryProps> = ({
  queryHistory,
  history,
  onQuerySelect,
  onSelectQuery,
  onClearHistory,
  onClear,
  isInSidebar = false
}) => {
  // Support both prop naming conventions
  const historyData = queryHistory || history || [];
  const handleSelectQuery = onQuerySelect || onSelectQuery || (() => {});
  const handleClear = onClearHistory || onClear || (() => {});

  if (historyData.length === 0) {
    return (
      <div className={isInSidebar ? '' : 'bg-white rounded-2xl shadow-xl p-6'}>
        {!isInSidebar && (
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <History className="w-5 h-5" />
            Recent Queries
          </h3>
        )}
        <p className={`text-sm ${isInSidebar ? 'text-gray-400' : 'text-gray-500'}`}>
          No queries yet. Try asking a question!
        </p>
      </div>
    );
  }

  return (
    <div className={isInSidebar ? '' : 'bg-white rounded-2xl shadow-xl p-6'}>
      {!isInSidebar && (
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
            <History className="w-5 h-5" />
            Recent Queries
          </h3>
          <button
            onClick={handleClear}
            className="text-red-600 hover:text-red-700 flex items-center gap-1 text-sm"
          >
            <Trash2 className="w-4 h-4" />
            Clear
          </button>
        </div>
      )}

      {isInSidebar && (
        <button
          onClick={handleClear}
          className="w-full mb-3 px-3 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg flex items-center justify-center gap-2 text-sm transition-colors"
        >
          <Trash2 className="w-4 h-4" />
          Clear All
        </button>
      )}

      <div className={`space-y-3 ${isInSidebar ? '' : 'max-h-96'} overflow-y-auto`}>
        {historyData.map((entry, idx) => (
          <div
            key={idx}
            onClick={() => handleSelectQuery(entry.query)}
            className={`p-3 rounded-lg cursor-pointer transition-all ${
              isInSidebar
                ? 'bg-gray-800 hover:bg-gray-700 border border-gray-700'
                : 'border border-gray-200 hover:bg-gray-50'
            }`}
          >
            <div className="flex items-start justify-between mb-1">
              <p className={`text-sm font-medium flex-1 ${
                isInSidebar ? 'text-gray-200' : 'text-gray-800'
              }`}>
                {entry.query}
              </p>
              <div className={`text-xs font-semibold ml-2 ${
                entry.result.confidence > 0.7 ? 'text-green-400' :
                entry.result.confidence > 0.4 ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {(entry.result.confidence * 100).toFixed(0)}%
              </div>
            </div>

            <div className={`flex items-center gap-2 text-xs ${
              isInSidebar ? 'text-gray-400' : 'text-gray-500'
            }`}>
              <Clock className="w-3 h-3" />
              {new Date(entry.timestamp).toLocaleString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default QueryHistory;