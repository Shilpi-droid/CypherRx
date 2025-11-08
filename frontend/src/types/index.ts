// src/types/index.ts
export interface Node {
    id: string;
    name: string;
    type: 'Drug' | 'Disease' | 'SideEffect' | 'Target' | 'Pathway';
  }
  
  export interface Edge {
    source: string;
    target: string;
    type: string;
    description?: string;
  }
  
  export interface Path {
    nodes: string[];
    node_types: string[];
    relationships: string[];
    score: number;
    evidence?: string[];
  }
  
  export interface QueryResult {
    answer: string;
    paths: Path[];
    confidence: number;
    timestamp?: string;
  }
  
  export interface HistoryEntry {
    query: string;
    result: QueryResult;
    timestamp: string;
  }