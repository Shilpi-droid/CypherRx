// src/hooks/useQuery.ts
import { useState } from 'react';
import { QueryResult, HistoryEntry } from '../types';
import { queryKnowledgeGraph } from '../services/api';

export const useQuery = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  const executeQuery = async (query: string) => {
    setLoading(true);
    setError(null);

    try {
      const data = await queryKnowledgeGraph(query);
      setResult(data);
      
      // Add to history
      const entry: HistoryEntry = {
        query,
        result: data,
        timestamp: new Date().toISOString()
      };
      setHistory(prev => [entry, ...prev].slice(0, 10)); // Keep last 10
      
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
  };

  return { executeQuery, loading, error, result, history, clearHistory };
};