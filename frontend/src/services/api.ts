// src/services/api.ts
import axios from 'axios';
import { FullGraphData, StreamEvent } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const getGraphStats = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/stats`, {
      timeout: 240000 // 4 minutes
    });
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw new Error('Failed to fetch graph statistics');
  }
};

export const getNodeNeighbors = async (name: string, type: string): Promise<FullGraphData> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/neighbors`, {
      params: { name, type },
      timeout: 30000
    });
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw new Error('Failed to fetch node neighbors');
  }
};

export const getFullGraph = async (): Promise<FullGraphData> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/graph`, {
      timeout: 60000
    });
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw new Error('Failed to fetch the full knowledge graph');
  }
};

// Streams beam-search progress via newline-delimited JSON so the UI can show
// reasoning paths as they're discovered instead of waiting for the full answer.
export const streamQuery = async (
  query: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, beam_width: 3, max_depth: 4 }),
    signal
  });

  if (!response.ok || !response.body) {
    throw new Error('Failed to start streaming query');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line));
    }
  }

  if (buffer.trim()) onEvent(JSON.parse(buffer));
};
