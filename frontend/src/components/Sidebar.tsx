import React from 'react';
import { Menu, X, History } from 'lucide-react';
import QueryHistory from './QueryHistory';
import { HistoryEntry } from '../types';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  queryHistory: HistoryEntry[];
  onQuerySelect: (query: string) => void;
  onClearHistory: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onToggle,
  queryHistory,
  onQuerySelect,
  onClearHistory,
}) => {
  return (
    <>
      {/* Backdrop for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar */}
      <div
        className={`fixed top-0 left-0 h-full bg-gray-900 text-white z-50 transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        } w-80 max-w-[85vw] flex flex-col shadow-2xl`}
      >
        {/* Sidebar Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <History className="w-5 h-5" />
            <h2 className="text-lg font-semibold">Recent Queries</h2>
          </div>
          <button
            onClick={onToggle}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Sidebar Content */}
        <div className="flex-1 overflow-y-auto p-4">
          <QueryHistory
            queryHistory={queryHistory}
            onQuerySelect={onQuerySelect}
            onClearHistory={onClearHistory}
            isInSidebar={true}
          />
        </div>
      </div>

      {/* Toggle Button (when sidebar is closed) */}
      {!isOpen && (
        <button
          onClick={onToggle}
          className="fixed top-4 left-4 z-40 p-3 bg-white/90 backdrop-blur-sm rounded-lg shadow-lg hover:shadow-xl transition-all hover:scale-105 hover:bg-white"
          aria-label="Open sidebar"
        >
          <Menu className="w-6 h-6 text-gray-800" />
        </button>
      )}
    </>
  );
};

export default Sidebar;
