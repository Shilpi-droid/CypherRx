// src/services/api.ts
import axios from 'axios';
import { QueryResult } from '../types';

const API_BASE_URL = 'https://cypherrx-backend.vercel.app';


export const queryKnowledgeGraph = async (query: string): Promise<QueryResult> => {
  try {
    const response = await axios.post(`${API_BASE_URL}/query`, {
      query,
      beam_width: 3,
      max_depth: 4
    }, {
      timeout: 2400000 // 4 minutes
    });
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw new Error('Failed to query knowledge graph');
  }
};

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