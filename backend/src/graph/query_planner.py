# src/reasoning/query_planner.py
"""
LLM-powered Query Planner
One call → finds best starting nodes in the graph
"""

import logging
import re
import time
from pydantic import BaseModel, Field, ValidationError
try:
    from backend.src.utils.llm_config import llm
    from backend.src.graph.schema_inspector import GraphSchemaInspector
    from backend.src.graph.prompts import (
        ENTITY_MAPPINGS,
        DEFAULT_SCHEMA,
        get_query_planner_system_prompt,
        get_user_prompt,
        get_correction_prompt
    )
except ModuleNotFoundError:
    from src.utils.llm_config import llm
    from src.graph.schema_inspector import GraphSchemaInspector
    from src.graph.prompts import (
        ENTITY_MAPPINGS,
        DEFAULT_SCHEMA,
        get_query_planner_system_prompt,
        get_user_prompt,
        get_correction_prompt
    )

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Rate Limiting for LLM Calls
# ----------------------------------------------------------------------
_last_llm_call_time = None
_MIN_DELAY_BETWEEN_CALLS = 30  # seconds

def _rate_limited_llm_invoke(prompt_or_messages):
    """
    Wrapper for LLM invocation with rate limiting.

    Args:
        prompt_or_messages: Either a string prompt or list of messages

    Returns:
        LLM response
    """
    global _last_llm_call_time

    # Apply rate limiting delay
    if _last_llm_call_time is not None:
        elapsed = time.time() - _last_llm_call_time
        if elapsed < _MIN_DELAY_BETWEEN_CALLS:
            wait_time = _MIN_DELAY_BETWEEN_CALLS - elapsed
            logger.info(f"Rate limiting: waiting {wait_time:.1f}s before next LLM call")
            time.sleep(wait_time)

    # Make the LLM call
    response = llm.invoke(prompt_or_messages)

    _last_llm_call_time = time.time()
    return response


class QueryPlan(BaseModel):
    """
    Structured output from LLM
    """
    cypher: str = Field(
        ...,
        description="Valid Cypher query to retrieve starting node(s). "
                    "Must return `name` and `type` columns."
    )


class QueryPlanner:
    """
    Uses LLM to generate a safe, executable Cypher query
    for finding the best starting node in the medical KG.
    """

    def __init__(self, driver=None, database="neo4j"):
        """
        Initialize QueryPlanner with optional schema inspection.

        Args:
            driver: Neo4j driver instance (optional, for schema inspection)
            database: Database name to inspect
        """
        self.schema_inspector = None
        if driver:
            self.schema_inspector = GraphSchemaInspector(driver, database)
            logger.info("Schema inspector enabled for QueryPlanner")
        
        # Use entity mappings from prompts module
        self.entity_mappings = ENTITY_MAPPINGS
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize colloquial terms to formal medical terminology.
        
        Args:
            query (str): Original user query
            
        Returns:
            str: Query with normalized entity names
        """
        normalized = query
        query_lower = query.lower()
        
        # Sort by length (longest first) to handle multi-word phrases
        sorted_mappings = sorted(self.entity_mappings.items(), key=lambda x: len(x[0]), reverse=True)
        
        for colloquial, formal in sorted_mappings:
            # Case-insensitive replacement
            if colloquial in query_lower:
                # Find the actual case-sensitive position
                pattern = re.compile(re.escape(colloquial), re.IGNORECASE)
                normalized = pattern.sub(formal, normalized)
                logger.debug(f"Normalized: '{colloquial}' → '{formal}'")
        
        if normalized != query:
            logger.info(f"Query normalized: '{query}' → '{normalized}'")
        
        return normalized

    def _extract_cypher(self, text: str) -> str:
        """
        Extract Cypher query from LLM response.
        Handles markdown code blocks and plain text.
        """
        # Try to extract from markdown code blocks
        cypher_match = re.search(r'```(?:cypher)?\s*\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if cypher_match:
            return cypher_match.group(1).strip()
        
        # If no code block, return the text as-is (assume it's already Cypher)
        return text.strip()

    def plan(self, query: str, entities: dict = None) -> str:
        """
        Generate Cypher using structured output with schema awareness.

        Args:
            query (str): Natural language user question
            entities (dict, optional): Extracted entities from query
                {
                    "drugs": [...],
                    "conditions": [...],
                    "drug_classes": [...]
                }

        Returns:
            str: Clean Cypher query
        """
        # Normalize colloquial terms to formal medical terminology
        normalized_query = self._normalize_query(query)
        
        # Use empty dict if no entities provided
        if entities is None:
            entities = {"drugs": [], "conditions": [], "drug_classes": []}

        # Get actual schema if available
        schema_info = ""
        if self.schema_inspector:
            try:
                schema = self.schema_inspector.get_schema()
                schema_info = f"""
**ACTUAL DATABASE SCHEMA** (Use this information - it's from the real database):
{schema.to_summary()}
"""
                logger.info("Using real-time schema information for query planning")
            except Exception as e:
                logger.warning(f"Could not fetch schema: {e}, using default schema")

        # Fallback to default schema if inspector not available
        if not schema_info:
            schema_info = DEFAULT_SCHEMA

        # Generate system prompt using prompts module
        system_prompt = get_query_planner_system_prompt(schema_info)
        
        # Generate user prompt with entity context
        user_prompt = get_user_prompt(normalized_query, entities)

        # Import messages at the top of the function
        from langchain_core.messages import SystemMessage, HumanMessage

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = _rate_limited_llm_invoke(messages)
            response_text = response.content.strip()
            
            # Extract Cypher from response (handle markdown code blocks if present)
            cypher = self._extract_cypher(response_text)

            logger.info(f"LLM generated Cypher:\n{cypher}")

            # Basic safety check
            if not cypher.upper().startswith("MATCH"):
                raise ValueError("Cypher does not start with MATCH")

            # Schema validation if available
            if self.schema_inspector:
                validation = self.schema_inspector.validate_cypher_entities(cypher)
                if not validation["valid"]:
                    logger.warning(f"Schema validation issues found:")
                    for issue in validation["issues"]:
                        logger.warning(f"  - {issue}")
                    for suggestion in validation["suggestions"]:
                        logger.info(f"  [HINT] {suggestion}")

                    # Try to auto-correct and regenerate if we have suggestions
                    if validation["suggestions"]:
                        logger.info("Attempting to regenerate query with schema hints...")
                        
                        # Generate correction prompt using prompts module
                        correction_prompt = get_correction_prompt(
                            validation["issues"],
                            validation["suggestions"],
                            normalized_query
                        )
                        
                        correction_messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=user_prompt),
                            HumanMessage(content=correction_prompt)
                        ]

                        corrected_response = _rate_limited_llm_invoke(correction_messages)
                        cypher = self._extract_cypher(corrected_response.content.strip())
                        logger.info(f"Regenerated Cypher:\n{cypher}")

            # Validate using Pydantic model
            try:
                validated = QueryPlan(cypher=cypher)
                return validated.cypher
            except ValidationError as ve:
                logger.warning(f"Validation error: {ve}, using extracted Cypher anyway")
                return cypher

        except Exception as e:
            logger.error(f"QueryPlanner failed: {e}")
            # Safe fallback
            return """
            MATCH (n) 
            WHERE n.name IS NOT NULL 
            RETURN 'unknown' AS name, 'Unknown' AS type 
            LIMIT 0
            """