// src/App.tsx
import React, { useState } from 'react';
import { Activity, Github, Info } from 'lucide-react';
import { QueryInput } from './components/QueryInput';
import { AnswerDisplay } from './components/AnswerDisplay';
import { ReasoningPath } from './components/ReasoningPath';
import { ConfidenceGauge } from './components/ConfidenceGauge';
import { GraphVisualization } from './components/GraphVisualization';
import Sidebar from './components/Sidebar';
import { useQuery } from './hooks/useQuery';

function App() {
  const { executeQuery, loading, error, result, history, clearHistory } = useQuery();
  const [showAbout, setShowAbout] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const handleQuery = async (query: string) => {
    await executeQuery(query);
  };

  return (
    <div className="min-h-screen pb-12">
      {/* Sidebar */}
      <Sidebar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        queryHistory={history}
        onQuerySelect={handleQuery}
        onClearHistory={clearHistory}
      />

      {/* Header */}
      <header className="bg-white/10 backdrop-blur-md border-b border-white/20">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-white rounded-xl">
                <Activity className="w-8 h-8 text-primary" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">
                  Medical Knowledge Graph
                </h1>
                <p className="text-white/80 text-sm">
                  Multi-hop causal reasoning assistant
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <button
                onClick={() => setShowAbout(!showAbout)}
                className="p-2 text-white hover:bg-white/10 rounded-lg transition-all"
              >
                <Info className="w-6 h-6" />
              </button>
              
              <a
                href="https://github.com/yourusername/medical-kg"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 text-white hover:bg-white/10 rounded-lg transition-all"
              >
                <Github className="w-6 h-6" />
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* About Modal */}
      {showAbout && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl p-8 max-w-2xl max-h-[80vh] overflow-y-auto">
            <h2 className="text-2xl font-bold mb-4">About This Project</h2>
            <div className="space-y-4 text-gray-700">
              <p>
                <strong>Medical Knowledge Graph Assistant</strong> uses advanced causal reasoning 
                over a knowledge graph containing drugs, diseases, and biological mechanisms.
              </p>
              <div>
                <h3 className="font-semibold mb-2">Features:</h3>
                <ul className="list-disc list-inside space-y-1 text-sm">
                  <li>Multi-hop reasoning through medical relationships</li>
                  <li>Causal pathway discovery</li>
                  <li>Explainable AI with full reasoning paths</li>
                  <li>Interactive graph visualization</li>
                  <li>Confidence scoring</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Tech Stack:</h3>
                <p className="text-sm">
                  React + TypeScript, Neo4j, Python, Beam Search Algorithm, 
                  Sentence Transformers
                </p>
              </div>
              <p className="text-sm text-yellow-700 bg-yellow-50 p-3 rounded">
                ⚠️ <strong>Disclaimer:</strong> For educational purposes only. 
                Not a substitute for professional medical advice.
              </p>
            </div>
            <button
              onClick={() => setShowAbout(false)}
              className="mt-6 w-full py-2 bg-primary text-white rounded-lg hover:bg-blue-600"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        {/* Query Input */}
        <QueryInput onSubmit={handleQuery} loading={loading} />

        {/* Error Display */}
        {error && (
          <div className="max-w-4xl mx-auto mb-6 p-4 bg-red-100 border-l-4 border-red-500 text-red-700 rounded">
            <p className="font-semibold">Error</p>
            <p className="text-sm">{error}</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="max-w-6xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
              <div className="lg:col-span-2">
                <AnswerDisplay 
                  answer={result.answer} 
                  confidence={result.confidence} 
                />
              </div>
              <div>
                <ConfidenceGauge confidence={result.confidence} />
              </div>
            </div>

            <ReasoningPath paths={result.paths} />

            {result.paths.length > 0 && (
              <GraphVisualization paths={result.paths} />
            )}
          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="fixed bottom-0 left-0 right-0 bg-white/10 backdrop-blur-md border-t border-white/20">
        <div className="container mx-auto px-4 py-3">
          <p className="text-center text-white/80 text-sm">
            Built with React, Neo4j, and ❤️ | Portfolio Project 2025
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;

