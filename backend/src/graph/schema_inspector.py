# backend/src/graph/schema_inspector.py
"""
Graph Schema Inspector
Fetches and caches the actual Neo4j graph schema for intelligent query planning.
"""

import logging
from typing import Dict, List, Set, Any
from neo4j import Driver
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GraphSchema:
    """Represents the actual structure of the Neo4j knowledge graph."""
    node_labels: Set[str] = field(default_factory=set)
    relationship_types: Set[str] = field(default_factory=set)
    node_properties: Dict[str, Set[str]] = field(default_factory=dict)
    relationship_patterns: List[Dict[str, Any]] = field(default_factory=list)

    def to_summary(self) -> str:
        """Generate a human-readable summary of the schema."""
        summary = []
        summary.append("**Available Node Labels:**")
        summary.append(", ".join(sorted(self.node_labels)))

        summary.append("\n**Available Relationship Types:**")
        summary.append(", ".join(sorted(self.relationship_types)))

        summary.append("\n**Common Relationship Patterns:**")
        for pattern in self.relationship_patterns[:15]:  # Top 15 patterns
            summary.append(f"  ({pattern['start']}) -[:{pattern['type']}]-> ({pattern['end']})")

        summary.append("\n**Node Properties:**")
        for label, props in sorted(self.node_properties.items()):
            summary.append(f"  {label}: {', '.join(sorted(props))}")

        return "\n".join(summary)


class GraphSchemaInspector:
    """
    Inspects Neo4j database to understand actual schema.
    Provides this information to the LLM for smarter query generation.
    """

    def __init__(self, driver: Driver, database: str = "neo4j"):
        self.driver = driver
        self.database = database
        self._cached_schema: GraphSchema = None

    def get_schema(self, force_refresh: bool = False) -> GraphSchema:
        """
        Get the graph schema. Uses cached version unless force_refresh=True.

        Args:
            force_refresh: If True, re-fetch schema from database

        Returns:
            GraphSchema object with current database structure
        """
        if self._cached_schema is not None and not force_refresh:
            return self._cached_schema

        logger.info("Inspecting Neo4j database schema...")
        schema = GraphSchema()

        with self.driver.session(database=self.database) as session:
            # 1. Get all node labels
            result = session.run("CALL db.labels()")
            schema.node_labels = {record["label"] for record in result}
            logger.info(f"Found node labels: {schema.node_labels}")

            # 2. Get all relationship types
            result = session.run("CALL db.relationshipTypes()")
            schema.relationship_types = {record["relationshipType"] for record in result}
            logger.info(f"Found relationship types: {schema.relationship_types}")

            # 3. Get node properties for each label
            for label in schema.node_labels:
                try:
                    result = session.run(f"""
                        MATCH (n:{label})
                        UNWIND keys(n) AS key
                        RETURN DISTINCT key
                        LIMIT 100
                    """)
                    schema.node_properties[label] = {record["key"] for record in result}
                except Exception as e:
                    logger.warning(f"Could not fetch properties for {label}: {e}")
                    schema.node_properties[label] = set()

            # 4. Get common relationship patterns (which nodes connect via which relationships)
            try:
                result = session.run("""
                    MATCH (start)-[r]->(end)
                    WITH labels(start)[0] AS start_label,
                         type(r) AS rel_type,
                         labels(end)[0] AS end_label,
                         count(*) AS count
                    RETURN start_label, rel_type, end_label, count
                    ORDER BY count DESC
                    LIMIT 50
                """)

                for record in result:
                    schema.relationship_patterns.append({
                        "start": record["start_label"],
                        "type": record["rel_type"],
                        "end": record["end_label"],
                        "count": record["count"]
                    })
            except Exception as e:
                logger.warning(f"Could not fetch relationship patterns: {e}")

        self._cached_schema = schema
        logger.info(f"Schema inspection complete: {len(schema.node_labels)} labels, "
                   f"{len(schema.relationship_types)} relationship types, "
                   f"{len(schema.relationship_patterns)} patterns")

        return schema

    def validate_cypher_entities(self, cypher: str) -> Dict[str, Any]:
        """
        Validate if entities in Cypher query actually exist in the schema.

        Args:
            cypher: The Cypher query to validate

        Returns:
            Dict with validation results and suggestions
        """
        schema = self.get_schema()
        issues = []
        suggestions = []

        # Extract node labels from query (simple pattern matching)
        import re

        # Find node labels like (n:Label) or (:Label)
        label_pattern = r'\((?:\w+)?:(\w+)(?:\s|\{)'
        found_labels = set(re.findall(label_pattern, cypher))

        for label in found_labels:
            if label not in schema.node_labels:
                issues.append(f"Label '{label}' does not exist in database")
                # Find similar labels
                similar = self._find_similar(label, schema.node_labels)
                if similar:
                    suggestions.append(f"Did you mean '{similar[0]}' instead of '{label}'?")

        # Find relationship types like [:REL_TYPE]
        rel_pattern = r'\[:(\w+)\]'
        found_rels = set(re.findall(rel_pattern, cypher))

        for rel in found_rels:
            if rel not in schema.relationship_types:
                issues.append(f"Relationship type '{rel}' does not exist in database")
                similar = self._find_similar(rel, schema.relationship_types)
                if similar:
                    suggestions.append(f"Did you mean '{similar[0]}' instead of '{rel}'?")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions
        }

    def _find_similar(self, term: str, candidates: Set[str], max_results: int = 3) -> List[str]:
        """Find similar terms using simple string similarity."""
        from difflib import get_close_matches
        return get_close_matches(term, candidates, n=max_results, cutoff=0.6)

    def get_sample_nodes(self, label: str, limit: int = 5) -> List[str]:
        """
        Get sample node names for a given label.
        Useful for debugging and examples.

        Args:
            label: Node label to sample from
            limit: Max number of samples

        Returns:
            List of node names
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                MATCH (n:{label})
                WHERE n.name IS NOT NULL
                RETURN n.name AS name
                LIMIT {limit}
            """)
            return [record["name"] for record in result]
