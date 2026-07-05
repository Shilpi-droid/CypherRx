// src/hooks/useQuery.ts
import { useState } from 'react';
import { QueryResult, HistoryEntry } from '../types';
import { streamQuery } from '../services/api';

export interface QueryProgress {
  depth: number;
  maxDepth: number;
}

export const useQuery = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [progress, setProgress] = useState<QueryProgress | null>(null);

  const executeQuery = async (query: string) => {
    setLoading(true);
    setError(null);
    setProgress(null);
    setResult(null);

    let finalResult: QueryResult | null = null;

    try {
      await streamQuery(query, event => {
        if (event.type === 'depth') {
          setProgress({ depth: event.depth, maxDepth: event.max_depth });
          setResult(prev => ({
            answer: prev?.answer ?? '',
            paths: event.paths,
            confidence: event.confidence
          }));
        } else if (event.type === 'final') {
          setProgress(null);
          finalResult = {
            answer: event.answer,
            paths: event.paths,
            confidence: event.confidence
          };
          setResult(finalResult);

          const entry: HistoryEntry = {
            query,
            result: finalResult,
            timestamp: new Date().toISOString()
          };
          setHistory(prev => [entry, ...prev].slice(0, 10)); // Keep last 10
        } else if (event.type === 'error') {
          setError(event.error);
        }
      });

      return finalResult;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
      setProgress(null);
    }
  };

  const clearHistory = () => {
    setHistory([]);
  };

  return { executeQuery, loading, error, result, history, clearHistory, progress };
};
