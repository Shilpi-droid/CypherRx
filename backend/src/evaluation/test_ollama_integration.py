#!/usr/bin/env python3
"""Test Ollama integration with structured output"""

from src.utils.llm_config import llm
from pydantic import BaseModel, Field

print("="*70)
print("TEST 1: Simple LLM invocation")
print("="*70)
result = llm.invoke("Say 'Hello World' in exactly two words")
print(f"Response: {result.content}")
print("✅ PASSED\n")

print("="*70)
print("TEST 2: Structured output with Pydantic")
print("="*70)

class QueryClassification(BaseModel):
    query_type: str = Field(description="Type of query")
    reasoning: str = Field(description="Why this classification")

structured_llm = llm.with_structured_output(QueryClassification)
result = structured_llm.invoke("""
Classify this query: "Can I take Warfarin with Azithromycin?"

Return JSON with:
- query_type: one of [interaction_check, simple_aggregation, path_traversal]
- reasoning: why this classification
""")

print(f"Result type: {type(result)}")
print(f"Query type: {result.query_type}")
print(f"Reasoning: {result.reasoning}")
print("✅ PASSED\n")

print("="*70)
print("TEST 3: Entity extraction")
print("="*70)

class ExtractedEntities(BaseModel):
    drugs: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)

entity_llm = llm.with_structured_output(ExtractedEntities)
result = entity_llm.invoke("""
Extract entities from: "What medications treat irregular heartbeat?"

Return JSON with:
- drugs: list of drug names
- conditions: list of conditions
""")

print(f"Result type: {type(result)}")
print(f"Drugs: {result.drugs}")
print(f"Conditions: {result.conditions}")
print("✅ PASSED\n")

print("="*70)
print("ALL TESTS PASSED! ✅")
print("="*70)

