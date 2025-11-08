// src/components/QueryInput.tsx
import React, { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';

interface QueryInputProps {
  onSubmit: (query: string) => void;
  loading: boolean;
}

const EXAMPLE_QUERIES = [
  "What are the side effects of Aspirin?",
  "What drugs treat Type 2 Diabetes?",
  "What medications are safe for kidney disease patients?",
  "Which hypertension drugs don't cause fatigue?",
];

export const QueryInput: React.FC<QueryInputProps> = ({ onSubmit, loading }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSubmit(query.trim());
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto mb-8">
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a medical question..."
            className="w-full px-6 py-4 pr-14 text-lg rounded-2xl shadow-xl border-2 border-gray-200 focus:border-primary focus:outline-none transition-all"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="absolute right-2 top-2 p-2 bg-primary text-white rounded-xl hover:bg-blue-600 disabled:bg-gray-400 transition-all"
          >
            {loading ? (
              <Loader2 className="w-6 h-6 animate-spin" />
            ) : (
              <Search className="w-6 h-6" />
            )}
          </button>
        </div>
      </form>

      {/* Example queries */}
      <div className="mt-4 flex flex-wrap gap-2">
        <span className="text-white text-sm font-medium">Try:</span>
        {EXAMPLE_QUERIES.map((example, idx) => (
          <button
            key={idx}
            onClick={() => setQuery(example)}
            className="text-xs bg-white/20 backdrop-blur-sm text-white px-3 py-1 rounded-full hover:bg-white/30 transition-all"
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
};