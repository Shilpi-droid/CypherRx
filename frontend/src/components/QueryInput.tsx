// src/components/QueryInput.tsx
import React, { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';

interface QueryInputProps {
  onSubmit: (query: string) => void;
  loading: boolean;
}

const EXAMPLE_QUERIES = [
  "I'm on Warfarin and need an antibiotic for a UTI. Which one is safe?",
  "I'm on Apixaban and have chronic kidney disease. Which blood thinner is safest?",
  "I take Metformin and Simvastatin. I have a sinus infection. Can I take Amoxicillin?",
  "I'm on Warfarin for DVT. I need pain relief. Can I take Ibuprofen?",
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
            className="absolute right-2 top-2 p-2 bg-primary text-white rounded-xl hover:bg-primary/90 disabled:bg-gray-400 transition-all"
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
        <span className="text-secondary text-sm font-medium">Try:</span>
        {EXAMPLE_QUERIES.map((example, idx) => (
          <button
            key={idx}
            onClick={() => setQuery(example)}
            className="text-xs bg-green-100 text-green-800 px-3 py-1 rounded-full hover:bg-green-200 transition-all"
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
};