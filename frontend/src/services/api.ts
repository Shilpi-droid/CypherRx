// src/services/api.ts
import axios from 'axios';
import { QueryResult } from '../types';

const API_BASE_URL = (process.env.REACT_APP_API_URL as string) || 'http://localhost:8000';

export const queryKnowledgeGraph = async (query: string): Promise<QueryResult> => {
  try {
    const response = await axios.post(`${API_BASE_URL}/query`, {
      query,
      beam_width: 5,
      max_depth: 4
    });
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw new Error('Failed to query knowledge graph');
  }
};

export const getGraphStats = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/stats`);
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw new Error('Failed to fetch graph statistics');
  }
};