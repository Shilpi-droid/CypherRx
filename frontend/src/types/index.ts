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

  export interface FullGraphNode {
    id: string;
    name: string;
    type: string;
  }

  export interface FullGraphLink {
    source: string;
    target: string;
    relationship: string;
  }

  export interface FullGraphData {
    nodes: FullGraphNode[];
    links: FullGraphLink[];
  }

  export type StreamEvent =
    | { type: 'starting_nodes'; nodes: { name: string; type: string }[] }
    | { type: 'depth'; depth: number; max_depth: number; paths: Path[]; confidence: number }
    | { type: 'final'; answer: string; paths: Path[]; confidence: number; query_type?: string }
    | { type: 'error'; error: string };