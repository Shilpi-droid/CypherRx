# src/reasoning/beam_searcher.py
"""
Think-on-Graph Beam Search Engine with LLM Guidance
Uses AzureChatOpenAI (llm) + Neo4j KG
"""

import os
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from neo4j import GraphDatabase
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# === Your LLM instance (shared across project) ===
try:
    from backend.src.utils.llm_config import llm
    from backend.src.graph.llm_agent import LLMAgent
    from backend.src.graph.query_planner import QueryPlanner
    from backend.src.graph.prompts import (
        get_query_classification_prompt,
        get_entity_normalization_prompt,
        get_entity_extraction_prompt,
        get_interaction_explanation_prompt,
        get_no_interaction_explanation_prompt,
        get_simple_aggregation_answer_prompt,
        get_filtered_aggregation_answer_prompt,
        get_intent_analysis_prompt,
        get_drug_class_interaction_answer_prompt,
        get_no_drug_class_interaction_answer_prompt,
        get_path_based_answer_prompt,
        get_cannot_answer_message_prompt,
        get_cypher_generation_prompt,
        get_query_intent_detection_prompt,
    )
except ModuleNotFoundError:
    # If running from backend directory
    from src.utils.llm_config import llm
    from src.graph.llm_agent import LLMAgent
    from src.graph.query_planner import QueryPlanner
    from src.graph.prompts import (
        get_query_classification_prompt,
        get_entity_normalization_prompt,
        get_entity_extraction_prompt,
        get_interaction_explanation_prompt,
        get_no_interaction_explanation_prompt,
        get_simple_aggregation_answer_prompt,
        get_filtered_aggregation_answer_prompt,
        get_intent_analysis_prompt,
        get_drug_class_interaction_answer_prompt,
        get_no_drug_class_interaction_answer_prompt,
        get_path_based_answer_prompt,
        get_cannot_answer_message_prompt,
        get_cypher_generation_prompt,
        get_query_intent_detection_prompt,
    )
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Pydantic Models for Structured Output
# ----------------------------------------------------------------------
class ExtractedEntities(BaseModel):
    """Structured output for entity extraction"""
    drugs: List[str] = Field(default_factory=list, description="List of drug names mentioned")
    conditions: List[str] = Field(default_factory=list, description="List of medical conditions mentioned")
    drug_classes: List[str] = Field(default_factory=list, description="List of drug classes mentioned")

class QueryClassification(BaseModel):
    """Structured output for query classification"""
    query_type: str = Field(default="simple_aggregation", description="One of: interaction_check, filtered_aggregation, simple_aggregation, path_traversal")
    reasoning: str = Field(default="Unable to classify", description="Brief explanation of why this classification was chosen")

class QueryIntent(BaseModel):
    """Structured output for query intent detection"""
    is_avoid_query: bool = Field(default=False, description="True if query asks which drugs to avoid/unsafe, False if asking for safe drugs")
    intent_type: str = Field(default="safe", description="One of: 'avoid' (find contraindicated), 'safe' (find safe drugs), 'neutral' (just listing)")
    reasoning: str = Field(default="", description="Brief explanation of the intent")


# ----------------------------------------------------------------------
# 1. Path Dataclass
# ----------------------------------------------------------------------
@dataclass
class Path:
    nodes: List[str]
    node_types: List[str]
    relationships: List[str]
    score: float = 0.0
    evidence: List[str] = None
    llm_reasoning: List[str] = None  # LLM's justification per step

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []
        if self.llm_reasoning is None:
            self.llm_reasoning = []

    def length(self) -> int:
        return len(self.nodes)

    def last_node(self) -> str:
        return self.nodes[-1] if self.nodes else None

    def last_node_type(self) -> str:
        return self.node_types[-1] if self.node_types else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "node_types": self.node_types,
            "relationships": self.relationships,
            "score": self.score,
            "evidence": self.evidence,
            "llm_reasoning": self.llm_reasoning,
        }

class BeamSearchReasoner:
    def __init__(self, beam_width: int = 3, max_depth: int = 4):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.llm_agent = LLMAgent(max_choices=beam_width)

        # Neo4j connection
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("DATABASE_NAME", "neo4j")

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

        # Initialize QueryPlanner with driver for schema-aware query generation
        self.query_planner = QueryPlanner(driver=self.driver, database=self.database)
        
        # Cache for database entities (to avoid repeated queries)
        self._entity_cache = None

        logger.info(f"BeamSearchReasoner initialized (K={beam_width}, depth={max_depth})")

    # ------------------------------------------------------------------
    # Entity Normalization with LLM
    # ------------------------------------------------------------------
    def get_database_entities(self) -> Dict[str, List[str]]:
        """Fetch actual entity names from the database for semantic matching (with caching)"""
        
        # Return cached entities if available
        if self._entity_cache is not None:
            return self._entity_cache
        
        drugs = []
        conditions = []
        
        try:
            with self.driver.session(database=self.database) as session:
                # Get all drug names
                drug_result = session.run("MATCH (d:Drug) RETURN d.name AS name ORDER BY name")
                drugs = [record["name"] for record in drug_result]
                
                # Get all condition names
                cond_result = session.run("MATCH (c:Condition) RETURN c.name AS name ORDER BY name")
                conditions = [record["name"] for record in cond_result]
                
            logger.info(f"Fetched {len(drugs)} drugs and {len(conditions)} conditions from database")
            
            # Cache the results
            self._entity_cache = {"drugs": drugs, "conditions": conditions}
            
        except Exception as e:
            logger.warning(f"Failed to fetch database entities: {e}")
            self._entity_cache = {"drugs": [], "conditions": []}
        
        return self._entity_cache
    
    def normalize_entity_with_llm(self, entity: str, entity_type: str, database_entities: List[str]) -> str:
        """Use LLM to semantically match entity to database names"""
        
        if not entity or not database_entities:
            return entity
        
        # Quick exact match first (case-insensitive)
        for db_entity in database_entities:
            if entity.lower() == db_entity.lower():
                return db_entity
        
        # Use LLM for semantic matching
        prompt = get_entity_normalization_prompt(entity, entity_type, database_entities)

        try:
            response = llm.invoke(prompt)
            matched = response.content.strip()
            
            # Verify the matched term is actually in the database
            if matched in database_entities:
                logger.info(f"LLM matched: '{entity}' → '{matched}'")
                return matched
            elif matched == "NO_MATCH":
                logger.info(f"No database match for: '{entity}'")
                return entity
            else:
                # LLM returned something not in list, try fuzzy match
                matched_lower = matched.lower()
                for db_entity in database_entities:
                    if matched_lower == db_entity.lower():
                        logger.info(f"LLM matched (case-insensitive): '{entity}' → '{db_entity}'")
                        return db_entity
                
                logger.warning(f"LLM returned invalid match '{matched}' for '{entity}'")
                return entity
                
        except Exception as e:
            logger.warning(f"LLM entity normalization failed: {e}")
            
            # Fallback to simple fuzzy matching
            entity_lower = entity.lower()
            for db_entity in database_entities:
                if entity_lower in db_entity.lower() or db_entity.lower() in entity_lower:
                    logger.info(f"Fuzzy fallback match: '{entity}' → '{db_entity}'")
                    return db_entity
            
            return entity

    def normalize_entities(self, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Normalize all entities using LLM-based semantic matching"""
        
        # Get actual database entities
        db_entities = self.get_database_entities()
        
        normalized = {
            "drugs": [],
            "conditions": [],
            "drug_classes": entities.get("drug_classes", [])  # Don't normalize classes
        }
        
        # Normalize drugs
        for drug in entities.get("drugs", []):
            if drug:
                normalized_drug = self.normalize_entity_with_llm(drug, "drug", db_entities["drugs"])
                normalized["drugs"].append(normalized_drug)
        
        # Normalize conditions
        for condition in entities.get("conditions", []):
            if condition:
                normalized_condition = self.normalize_entity_with_llm(condition, "condition", db_entities["conditions"])
                normalized["conditions"].append(normalized_condition)
        
        return normalized

    # ------------------------------------------------------------------
    # Query Classification (LLM-based with Structured Output)
    # ------------------------------------------------------------------
    def classify_query_type(self, query: str) -> str:
        """Use LLM with structured output to classify query type"""
        
        prompt = get_query_classification_prompt(query)

        try:
            # Use structured output with Pydantic
            structured_llm = llm.with_structured_output(QueryClassification)
            result = structured_llm.invoke(prompt)
            
            logger.info(f"Query classification: {result.query_type} (reasoning: {result.reasoning})")
            return result.query_type
            
        except Exception as e:
            logger.warning(f"Structured classification failed: {e}. Using fallback parsing")
            
            # Fallback to text parsing
            try:
                response = llm.invoke(prompt)
                classification = response.content.strip().lower()
                
                if "interaction" in classification:
                    return "interaction_check"
                elif "filtered_aggregation" in classification:
                    return "filtered_aggregation"
                elif "simple_aggregation" in classification:
                    return "simple_aggregation"
                else:
                    return "path_traversal"
            except Exception as e2:
                logger.warning(f"Fallback classification failed: {e2}. Defaulting to path_traversal")
                return "path_traversal"

    # ------------------------------------------------------------------
    # Entity Extraction (LLM-enhanced)
    # ------------------------------------------------------------------
    def extract_entities_with_llm(self, query: str) -> Dict[str, List[str]]:
        """Use LLM with structured output to extract and normalize medical entities"""
        prompt = get_entity_extraction_prompt(query)

        try:
            # Use structured output with Pydantic
            structured_llm = llm.with_structured_output(ExtractedEntities)
            entities = structured_llm.invoke(prompt)
            
            # Validate with Pydantic
            try:
                validated_entities = ExtractedEntities.model_validate(entities.model_dump())
            except Exception as validation_error:
                logger.warning(f"Pydantic validation failed: {validation_error}. Using empty entities.")
                validated_entities = ExtractedEntities()
            
            # Convert Pydantic model to dict
            entities_dict = {
                "drugs": validated_entities.drugs,
                "conditions": validated_entities.conditions,
                "drug_classes": validated_entities.drug_classes
            }
            
            logger.info(f"Extracted entities (raw): {entities_dict}")
            
            # Normalize entities to match graph
            normalized = self.normalize_entities(entities_dict)
            logger.info(f"Normalized entities: {normalized}")
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Structured entity extraction failed: {e}. Using fallback")
            
            # Fallback: try to parse JSON from text response
            try:
                response = llm.invoke(prompt)
                content = response.content.strip()
                
                # Check if content is empty or just whitespace
                if not content or content == "{}":
                    logger.warning("Empty response from LLM fallback")
                    return {"drugs": [], "conditions": [], "drug_classes": []}
                
                import json
                
                # Try to clean up common JSON issues
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                entities = json.loads(content)
                normalized = self.normalize_entities(entities)
                logger.info(f"Extracted and normalized entities (fallback): {normalized}")
                return normalized
                
            except json.JSONDecodeError as e2:
                logger.warning(f"Fallback JSON parsing failed: {e2}. Response was: {response.content[:100]}")
                return {"drugs": [], "conditions": [], "drug_classes": []}
            except Exception as e2:
                logger.warning(f"Fallback entity extraction failed: {e2}")
                return {"drugs": [], "conditions": [], "drug_classes": []}

    def extract_keywords(self, query: str) -> List[str]:
        """Legacy keyword extraction - kept for fallback"""
        query_lower = query.lower()
        known_terms = [
            # Drugs
            "metformin", "insulin", "warfarin", "apixaban", "amoxicillin", "lisinopril",
            "atorvastatin", "sitagliptin", "semaglutide", "levothyroxine",
            # Conditions
            "diabetes", "hypertension", "pneumonia", "kidney disease", "atrial fibrillation",
            "depression", "asthma", "gout", "pregnancy", "uti"
        ]
        found = [term.title() for term in known_terms if term in query_lower]
        if not found:
            words = [w.title() for w in query_lower.split() if len(w) > 4]
            found = words[:3]
        return found

    def detect_query_intent(self, query: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Use LLM to detect if query is asking for 'avoid' or 'safe' drugs"""
        # Use the function already imported at the top of the file
        prompt = get_query_intent_detection_prompt(query, entities)
        
        try:
            # Use structured output with Pydantic
            structured_llm = llm.with_structured_output(QueryIntent)
            result = structured_llm.invoke(prompt)
            
            logger.info(f"Query intent detection: is_avoid={result.is_avoid_query}, type={result.intent_type}, reasoning={result.reasoning}")
            return {
                "is_avoid_query": result.is_avoid_query,
                "intent_type": result.intent_type,
                "reasoning": result.reasoning
            }
        except Exception as e:
            logger.warning(f"LLM intent detection failed: {e}. Falling back to keyword detection")
            # Fallback to keyword-based detection
            is_avoid = any(word in query.lower() for word in ["avoid", "unsafe", "should not", "shouldn't", "cannot take", "can't take", "should i avoid", "which should i avoid"])
            return {
                "is_avoid_query": is_avoid,
                "intent_type": "avoid" if is_avoid else "safe",
                "reasoning": "Fallback keyword detection"
            }
    

    # ------------------------------------------------------------------
    # Find starting nodes
    # ------------------------------------------------------------------
    def find_starting_nodes(self, query: str) -> List[Tuple[str, str]]:
        logger.info(f"Planning starting nodes for: {query}")
        
        # Extract entities to help guide Cypher generation
        entities = self.extract_entities_with_llm(query)
        logger.info(f"Entities for query planning: {entities}")
        
        # Generate Cypher with entity context
        cypher = self.query_planner.plan(query, entities)

        if "none" in cypher.lower() or "error" in cypher.lower():
            logger.warning("LLM found no relevant starting node")
            return []

        starting_nodes = []
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher)
                for record in result:
                    name = record.get("name")
                    ntype = record.get("type", "Unknown")
                    if name and name not in ["none", "error"]:
                        starting_nodes.append((name, ntype))
        except Exception as e:
            logger.error(f"Cypher failed: {e}\nQuery: {cypher}")
            return []

        logger.info(f"Found {len(starting_nodes)} starting nodes: {starting_nodes}")
        return starting_nodes

    # ------------------------------------------------------------------
    # LLM-powered interaction explanations
    # ------------------------------------------------------------------
    def _explain_interaction_with_llm(self, drug1: str, drug2: str, severity: str, 
                                      description: str, query: str) -> str:
        """Use LLM to generate detailed explanation of drug interaction"""
        
        prompt = get_interaction_explanation_prompt(drug1, drug2, severity, description, query)

        try:
            response = llm.invoke(prompt)
            answer = response.content.strip()
            logger.info("Generated interaction explanation using LLM")
            return answer
        except Exception as e:
            logger.warning(f"LLM interaction explanation failed: {e}, using fallback")
            severity_emoji = "🔴" if severity == "MAJOR" else "🟡"
            return f"{severity_emoji} **WARNING: {severity} Drug Interaction**\n\n{drug1} and {drug2} interact.\n\n**Details:** {description}\n\n⚠️ Consult your healthcare provider before taking these medications together."
    
    def _explain_no_interaction_with_llm(self, drug1: str, drug2: str, query: str) -> str:
        """Use LLM to generate reassuring response when no interaction found"""
        
        prompt = get_no_interaction_explanation_prompt(drug1, drug2, query)

        try:
            response = llm.invoke(prompt)
            answer = response.content.strip()
            logger.info("Generated no-interaction response using LLM")
            return answer
        except Exception as e:
            logger.warning(f"LLM no-interaction explanation failed: {e}, using fallback")
            return f"✅ No known interaction found between {drug1} and {drug2} in our database.\n\nHowever, always consult your healthcare provider about drug combinations."

    # ------------------------------------------------------------------
    # LLM-powered answer synthesis for aggregation results
    # ------------------------------------------------------------------
    def _synthesize_aggregation_answer(self, query: str, condition: str, results: List[Dict], 
                                      query_type: str, safety_condition: str = None, is_avoid_query: bool = False,
                                      execution_context: Dict[str, Any] = None) -> str:
        """Use LLM to generate natural, informative answer from aggregation results
        
        Args:
            query: Original user query
            condition: Condition being treated (or safety condition for avoid queries)
            results: Query results
            query_type: Type of query (simple_list, filtered_list)
            safety_condition: Safety constraint condition
            is_avoid_query: Whether this is an "avoid" query
            execution_context: Context from earlier execution steps (intent, Cypher query, etc.)
        """
        
        # Format results for LLM
        if query_type == "simple_list":
            drugs_info = []
            for r in results:
                drugs_info.append(f"• {r['drug']} ({r['drug_class']})")
            drugs_text = "\n".join(drugs_info)
            
            prompt = get_simple_aggregation_answer_prompt(query, condition, drugs_text)
        
        elif query_type == "filtered_list":
            if is_avoid_query:
                # For "avoid" queries: results are contraindicated (to avoid) or require adjustment (caution)
                contraindicated_drugs = [r for r in results if not r.get("is_caution")]
                caution_drugs = [r for r in results if r.get("is_caution")]
                
                avoid_text = "\n".join([f"• {r['drug']} ({r['drug_class']})" for r in contraindicated_drugs]) if contraindicated_drugs else "None"
                caution_text = "\n".join([f"• {r['drug']} ({r['drug_class']})" for r in caution_drugs]) if caution_drugs else "None"
                
                prompt = get_filtered_aggregation_answer_prompt(
                    query, condition, safety_condition, avoid_text, caution_text, 
                    is_avoid_query=True, execution_context=execution_context
                )
            else:
                # For "safe" queries: results are safe or require adjustment
                safe_drugs = [r for r in results if not r.get("needs_adjustment")]
                adjust_drugs = [r for r in results if r.get("needs_adjustment")]
                
                safe_text = "\n".join([f"• {r['drug']} ({r['drug_class']})" for r in safe_drugs]) if safe_drugs else "None"
                adjust_text = "\n".join([f"• {r['drug']} ({r['drug_class']})" for r in adjust_drugs]) if adjust_drugs else "None"
                
                prompt = get_filtered_aggregation_answer_prompt(
                    query, condition, safety_condition, safe_text, adjust_text,
                    is_avoid_query=False, execution_context=execution_context
                )
        
        else:
            # Fallback
            return "Results found: " + str(len(results)) + " medications"
        
        try:
            response = llm.invoke(prompt)
            answer = response.content.strip()
            logger.info("Synthesized aggregation answer using LLM")
            return answer
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}, using fallback formatting")
            # Fallback to simple formatting
            drug_list = [f"{r['drug']} ({r['drug_class']})" for r in results]
            return f"**Medications for {condition}:**\n\n" + "\n".join(f"• {d}" for d in drug_list)

    # ------------------------------------------------------------------
    # Aggregation Query Execution (LLM-guided)
    # ------------------------------------------------------------------
    def execute_aggregation_query(self, query: str, query_type: str) -> Dict[str, Any]:
        """Handle queries that need filtering/aggregation using LLM to understand intent"""
        logger.info(f"Executing {query_type} query: {query}")
        
        # Build execution context to pass to answer synthesis
        execution_context = {
            "original_query": query,
            "entities": None,  # Will be filled below
            "intent_analysis": None,  # Will be filled below
            "query_intent": None,  # Will be filled later
            "cypher_query": None,  # Will be filled later
            "cypher_parameters": None,  # Will be filled later
            "query_type": query_type
        }
        
        # Extract entities using LLM
        entities = self.extract_entities_with_llm(query)
        execution_context["entities"] = entities
        
        # Use LLM to determine query intent and constraints
        intent_prompt = get_intent_analysis_prompt(query, entities)

        try:
            response = llm.invoke(intent_prompt)
            intent = response.content.strip()
            logger.info(f"Query intent: {intent}")
            execution_context["intent_analysis"] = intent
        except Exception as e:
            logger.warning(f"Intent analysis failed: {e}")
            intent = "Unknown intent"
            execution_context["intent_analysis"] = intent
        
        # Build and execute Cypher query based on type
        results = []
        
        if query_type == "simple_aggregation":
            # Simple: What treats X?
            conditions = entities.get('conditions', [])
            if conditions:
                treat_condition = conditions[0]
                cypher = """
                MATCH (d:Drug)-[:TREATS]->(c:Condition {name: $condition})
                RETURN d.name AS drug, 
                       d.class AS drug_class,
                       d.pregnancy_category AS pregnancy_category
                ORDER BY d.name
                """
                
                try:
                    with self.driver.session(database=self.database) as session:
                        result = session.run(cypher, condition=treat_condition)
                        results = [dict(record) for record in result]
                    
                    if results:
                        drug_list = [f"{r['drug']} ({r['drug_class']})" for r in results]
                        
                        # Use LLM to generate a more informative answer
                        answer = self._synthesize_aggregation_answer(
                            query=query,
                            condition=treat_condition,
                            results=results,
                            query_type="simple_list"
                        )
                        
                        return {
                            "answer": answer,
                            "results": results,
                            "query_type": query_type,
                            "intent": intent,
                            "confidence": 0.95
                        }
                except Exception as e:
                    logger.error(f"Cypher execution failed: {e}")
        
        elif query_type == "filtered_aggregation":
            # Use LLM to generate Cypher query based on intent (instead of hardcoded patterns)
            try:
                # Normalize drug classes BEFORE passing to LLM (so prompt uses correct class names)
                normalized_entities = entities.copy()
                if 'drug_classes' in normalized_entities and normalized_entities['drug_classes']:
                    drug_classes = normalized_entities['drug_classes']
                    normalized_classes = []
                    for dc in drug_classes:
                        dc_lower = dc.lower().rstrip('s')
                        class_normalization = {
                            "blood thinner": "anticoagulant",
                            "anticoagulant": "anticoagulant",
                            "antibiotic": "antibiotic",
                            "statin": "statin",
                            "ssri": "ssri",
                            "ace inhibitor": "ace inhibitor",
                        }
                        normalized = class_normalization.get(dc_lower, dc)
                        normalized_classes.append(normalized)
                    normalized_entities['drug_classes'] = normalized_classes
                    logger.info(f"Normalized drug classes: {drug_classes} → {normalized_classes}")
                
                # Get schema info for better query generation
                schema_info = ""
                if self.query_planner and self.query_planner.schema_inspector:
                    try:
                        schema = self.query_planner.schema_inspector.get_schema()
                        schema_info = f"\n**Database Schema**:\n{schema.to_summary()}\n"
                    except:
                        pass
                
                # Generate Cypher query using LLM
                cypher_prompt = get_cypher_generation_prompt(query, intent, normalized_entities, schema_info)
                logger.info("Generating Cypher query using LLM for filtered aggregation")
                
                cypher_response = llm.invoke(cypher_prompt)
                cypher = cypher_response.content.strip()
                
                # Clean up Cypher (remove markdown code blocks if present)
                if '```cypher' in cypher:
                    cypher = cypher.split('```cypher')[1].split('```')[0].strip()
                elif '```' in cypher:
                    cypher = cypher.split('```')[1].split('```')[0].strip()
                
                # Sanitize: Remove any non-Cypher lines (like "Parameters:", explanations, etc.)
                lines = []
                for line in cypher.splitlines():
                    line_stripped = line.strip()
                    # Skip empty lines, comment lines, and parameter documentation lines
                    if (line_stripped and 
                        not line_stripped.lower().startswith("parameters:") and
                        not line_stripped.lower().startswith("parameter:") and
                        not line_stripped.startswith("//") and
                        not line_stripped.startswith("--") and
                        not line_stripped.startswith("#")):
                        lines.append(line)
                
                cypher = "\n".join(lines).strip()
                logger.info(f"LLM generated Cypher (sanitized):\n{cypher}")
                
                # Store Cypher query in execution context
                execution_context["cypher_query"] = cypher
                
                # Extract parameters from the query
                import re
                param_pattern = r'\$(\w+)'
                params = set(re.findall(param_pattern, cypher))
                logger.info(f"Detected parameters: {params}")
                
                # Build parameter dictionary
                param_dict = {}
                drugs = entities.get('drugs', [])
                conditions = entities.get('conditions', [])
                drug_classes = entities.get('drug_classes', [])
                
                # Detect query intent using LLM
                intent_result = self.detect_query_intent(query, entities)
                is_avoid_query = intent_result["is_avoid_query"]
                execution_context["query_intent"] = intent_result
                
                # Map parameters to entity values with intelligent context-aware mapping
                # Drug class parameters - normalize to singular form
                if 'drug_class' in params or 'drug_classes' in params:
                    if drug_classes:
                        # Normalize drug class (remove plural, get stem)
                        drug_class_raw = drug_classes[0].lower().rstrip('s')
                        # Map common variations
                        class_normalization = {
                            "antibiotic": "antibiot",
                            "statin": "statin",
                            "blood thinner": "anticoagulant",
                            "anticoagulant": "anticoagulant",
                            "ssri": "ssri",
                            "ace inhibitor": "ace inhibitor",
                        }
                        drug_class_normalized = class_normalization.get(drug_class_raw, drug_class_raw)
                        param_dict['drug_class'] = drug_class_normalized
                        if 'drug_classes' in params:
                            param_dict['drug_classes'] = [drug_class_normalized]
                
                # Condition parameters - determine treat vs safety based on context
                if conditions:
                    # Key insight: When drug class is mentioned, the condition that needs treatment
                    # is usually the one mentioned with the drug class (e.g., "antibiotic for pneumonia")
                    # The other condition is usually a patient state (diabetes, pregnancy)
                    
                    if drug_classes and len(conditions) >= 2:
                        # We have drug class + multiple conditions
                        # Determine which condition is the "treat" condition
                        # Usually: treat condition is the one needing treatment (pneumonia, UTI, infection)
                        # Safety condition is patient state (diabetes, pregnancy, kidney disease)
                        
                        # Heuristic: conditions that typically need treatment
                        treat_indicators = ["pneumonia", "uti", "infection", "hypertension", "hyperlipidemia", "fibrillation", "atrial"]
                        # Conditions that are patient states
                        state_indicators = ["diabetes", "pregnancy", "kidney", "renal", "ckd"]
                        
                        treat_idx = 0
                        safety_idx = 1
                        
                        # Check which condition matches treat indicators
                        for i, cond in enumerate(conditions):
                            cond_lower = cond.lower()
                            if any(indicator in cond_lower for indicator in treat_indicators):
                                treat_idx = i
                                safety_idx = 1 - i  # The other one
                                break
                            elif any(indicator in cond_lower for indicator in state_indicators):
                                safety_idx = i
                                treat_idx = 1 - i  # The other one
                                break
                        
                        # Map parameters
                        if 'treat_condition' in params:
                            param_dict['treat_condition'] = conditions[treat_idx]
                        if 'safety_condition' in params:
                            param_dict['safety_condition'] = conditions[safety_idx]
                        if 'condition' in params and 'treat_condition' not in params:
                            param_dict['condition'] = conditions[treat_idx]
                    elif drug_classes and len(conditions) == 1:
                        # Drug class + single condition
                        if is_avoid_query:
                            # For "avoid" queries: single condition is the safety constraint
                            if 'safety_condition' in params:
                                param_dict['safety_condition'] = conditions[0]
                            if 'condition' in params and 'safety_condition' not in params:
                                param_dict['condition'] = conditions[0]
                        else:
                            # For "safe" queries: single condition is what we're treating
                            if 'condition' in params:
                                param_dict['condition'] = conditions[0]
                            if 'treat_condition' in params:
                                param_dict['treat_condition'] = conditions[0]
                    elif len(conditions) == 1:
                        # Single condition without drug class
                        if is_avoid_query:
                            # For "avoid" queries: single condition is the safety constraint
                            if 'safety_condition' in params:
                                param_dict['safety_condition'] = conditions[0]
                            if 'condition' in params and 'safety_condition' not in params:
                                param_dict['condition'] = conditions[0]
                        else:
                            # For "safe" queries: single condition is what we're treating
                            if 'condition' in params:
                                param_dict['condition'] = conditions[0]
                            if 'treat_condition' in params:
                                param_dict['treat_condition'] = conditions[0]
                    elif len(conditions) >= 2:
                        # Two conditions without drug class - first is treat, second is safety
                        if 'treat_condition' in params:
                            param_dict['treat_condition'] = conditions[0]
                        if 'safety_condition' in params:
                            param_dict['safety_condition'] = conditions[1]
                        if 'condition' in params and 'treat_condition' not in params:
                            param_dict['condition'] = conditions[0]
                
                # Drug constraint parameters
                if 'avoid_drug' in params:
                    if drugs:
                        param_dict['avoid_drug'] = drugs[0]
                if 'drug_name' in params and 'avoid_drug' not in param_dict:
                    if drugs:
                        param_dict['drug_name'] = drugs[0]
                
                # Validate required parameters are present
                missing_params = []
                for param in params:
                    if param not in param_dict:
                        missing_params.append(param)
                
                if missing_params:
                    logger.warning(f"Missing parameters: {missing_params}. Available entities: drugs={drugs}, conditions={conditions}, drug_classes={drug_classes}")
                    # Try to fill missing params with defaults or skip
                    # For now, log warning but continue
                
                # Only pass parameters that are actually in the query
                final_param_dict = {k: v for k, v in param_dict.items() if k in params}
                
                # Check for missing required parameters
                missing_params = params - set(final_param_dict.keys())
                if missing_params:
                    logger.warning(f"Missing required parameters: {missing_params}")
                    logger.warning(f"Available entities: drugs={drugs}, conditions={conditions}, drug_classes={drug_classes}")
                
                logger.info(f"Executing Cypher with parameters: {final_param_dict}")
                logger.info(f"Required parameters: {params}, Provided: {list(final_param_dict.keys())}")
                
                # Store Cypher parameters in execution context
                execution_context["cypher_parameters"] = final_param_dict
                
                # Execute the query with retry logic
                max_retries = 2
                results = []
                
                for attempt in range(max_retries):
                    try:
                        with self.driver.session(database=self.database) as session:
                            result = session.run(cypher, **final_param_dict)
                            results = [dict(record) for record in result]
                            logger.info(f"Query returned {len(results)} result(s)")
                            
                            # Log first 5 results for debugging
                            if results:
                                first_5 = [r.get('drug', r.get('name', 'N/A')) for r in results[:5]]
                                logger.info(f"First 5 results: {first_5}")
                            break  # Success, exit retry loop
                    except Exception as e:
                        logger.error(f"Cypher execution failed (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.error(f"Failed Cypher:\n{cypher}")
                        logger.error(f"Failed parameters: {final_param_dict}")
                        
                        if attempt < max_retries - 1:
                            # Try to regenerate Cypher with error feedback
                            logger.info("Attempting to regenerate Cypher with error feedback...")
                            error_feedback = f"The previous Cypher query failed with error: {str(e)}\n\nPlease regenerate a simpler, correct Cypher query."
                            retry_prompt = get_cypher_generation_prompt(query, intent, entities, schema_info) + "\n\n" + error_feedback
                            
                            try:
                                retry_response = llm.invoke(retry_prompt)
                                cypher = retry_response.content.strip()
                                
                                # Sanitize again
                                if '```cypher' in cypher:
                                    cypher = cypher.split('```cypher')[1].split('```')[0].strip()
                                elif '```' in cypher:
                                    cypher = cypher.split('```')[1].split('```')[0].strip()
                                
                                # Sanitize: Remove any non-Cypher lines
                                lines = []
                                for line in cypher.splitlines():
                                    line_stripped = line.strip()
                                    if (line_stripped and 
                                        not line_stripped.lower().startswith("parameters:") and
                                        not line_stripped.lower().startswith("parameter:") and
                                        not line_stripped.startswith("//") and
                                        not line_stripped.startswith("--") and
                                        not line_stripped.startswith("#")):
                                        lines.append(line)
                                
                                cypher = "\n".join(lines).strip()
                                logger.info(f"Regenerated Cypher (attempt {attempt + 2}):\n{cypher}")
                                
                                # Update Cypher query in execution context
                                execution_context["cypher_query"] = cypher
                                
                                # Re-extract parameters and re-map
                                params = set(re.findall(param_pattern, cypher))
                                logger.info(f"Regenerated query parameters: {params}")
                                
                                # Re-map parameters (reuse the same mapping logic)
                                param_dict = {}
                                
                                # Drug class parameters
                                if 'drug_class' in params or 'drug_classes' in params:
                                    if drug_classes:
                                        drug_class_raw = drug_classes[0].lower().rstrip('s')
                                        class_normalization = {
                                            "antibiotic": "antibiot",
                                            "statin": "statin",
                                            "blood thinner": "anticoagulant",
                                            "anticoagulant": "anticoagulant",
                                            "ssri": "ssri",
                                            "ace inhibitor": "ace inhibitor",
                                        }
                                        drug_class_normalized = class_normalization.get(drug_class_raw, drug_class_raw)
                                        param_dict['drug_class'] = drug_class_normalized
                                        if 'drug_classes' in params:
                                            param_dict['drug_classes'] = [drug_class_normalized]
                                
                                # Condition parameters (reuse same logic)
                                if conditions:
                                    if drug_classes and len(conditions) >= 2:
                                        treat_indicators = ["pneumonia", "uti", "infection", "hypertension", "hyperlipidemia", "fibrillation", "atrial"]
                                        treat_idx = 0
                                        for i, cond in enumerate(conditions):
                                            cond_lower = cond.lower()
                                            if any(indicator in cond_lower for indicator in treat_indicators):
                                                treat_idx = i
                                                break
                                        
                                        if 'treat_condition' in params:
                                            param_dict['treat_condition'] = conditions[treat_idx]
                                        if 'safety_condition' in params:
                                            param_dict['safety_condition'] = conditions[1 - treat_idx]
                                        if 'condition' in params and 'treat_condition' not in params:
                                            param_dict['condition'] = conditions[treat_idx]
                                    elif drug_classes and len(conditions) == 1:
                                        # Detect intent for retry
                                        intent_result = self.detect_query_intent(query, entities)
                                        is_avoid_query_retry = intent_result["is_avoid_query"]
                                        
                                        if is_avoid_query_retry:
                                            # For "avoid" queries: single condition is the safety constraint
                                            if 'safety_condition' in params:
                                                param_dict['safety_condition'] = conditions[0]
                                            if 'condition' in params and 'safety_condition' not in params:
                                                param_dict['condition'] = conditions[0]
                                        else:
                                            # For "safe" queries: single condition is what we're treating
                                            if 'condition' in params:
                                                param_dict['condition'] = conditions[0]
                                            if 'treat_condition' in params:
                                                param_dict['treat_condition'] = conditions[0]
                                    elif len(conditions) == 1:
                                        # Detect intent for retry
                                        intent_result = self.detect_query_intent(query, entities)
                                        is_avoid_query_retry = intent_result["is_avoid_query"]
                                        
                                        if is_avoid_query_retry:
                                            # For "avoid" queries: single condition is the safety constraint
                                            if 'safety_condition' in params:
                                                param_dict['safety_condition'] = conditions[0]
                                            if 'condition' in params and 'safety_condition' not in params:
                                                param_dict['condition'] = conditions[0]
                                        else:
                                            # For "safe" queries: single condition is what we're treating
                                            if 'condition' in params:
                                                param_dict['condition'] = conditions[0]
                                            if 'treat_condition' in params:
                                                param_dict['treat_condition'] = conditions[0]
                                    elif len(conditions) >= 2:
                                        if 'treat_condition' in params:
                                            param_dict['treat_condition'] = conditions[0]
                                        if 'safety_condition' in params:
                                            param_dict['safety_condition'] = conditions[1]
                                        if 'condition' in params and 'treat_condition' not in params:
                                            param_dict['condition'] = conditions[0]
                                
                                # Drug constraint parameters
                                if 'avoid_drug' in params:
                                    if drugs:
                                        param_dict['avoid_drug'] = drugs[0]
                                if 'drug_name' in params and 'avoid_drug' not in param_dict:
                                    if drugs:
                                        param_dict['drug_name'] = drugs[0]
                                
                                # Update final_param_dict
                                final_param_dict = {k: v for k, v in param_dict.items() if k in params}
                                logger.info(f"Remapped parameters: {final_param_dict}")
                                
                                # Update Cypher parameters in execution context
                                execution_context["cypher_parameters"] = final_param_dict
                            except Exception as retry_error:
                                logger.error(f"Failed to regenerate Cypher: {retry_error}")
                                raise  # Give up after max retries
                        else:
                            # Last attempt failed
                            raise
                
                # Class compliance check: Verify results match requested drug class
                if results and drug_classes:
                    requested_class = normalized_entities.get('drug_classes', drug_classes)[0].lower() if 'drug_classes' in normalized_entities else drug_classes[0].lower()
                    # Normalize to stem for matching
                    class_stem = requested_class.rstrip('s')
                    if class_stem == "antibiotic":
                        class_stem = "antibiot"
                        class_matches = lambda dc: class_stem in dc
                    elif class_stem == "anticoagulant" or requested_class == "blood thinner":
                        # Blood thinners use multiple class names: DOAC, Vitamin K Antagonist, LMWH
                        class_matches = lambda dc: "doac" in dc or "vitamin k antagonist" in dc or "lmwh" in dc
                    else:
                        class_matches = lambda dc: class_stem in dc or dc in class_stem
                    
                    filtered_results = []
                    for r in results:
                        drug_class = r.get('drug_class', '').lower()
                        # Check if drug class matches
                        if class_matches(drug_class):
                            filtered_results.append(r)
                    
                    if len(filtered_results) < len(results):
                        logger.warning(f"Class compliance: Filtered {len(results)} -> {len(filtered_results)} results matching '{requested_class}'")
                        results = filtered_results
                
                # Detect query intent using LLM
                intent_result = self.detect_query_intent(query, entities)
                is_avoid_query = intent_result["is_avoid_query"]
                
                if not results and is_avoid_query and drug_classes and len(conditions) >= 1:
                    # Try fallback: Find drugs that require adjustment (use with caution) instead of contraindicated
                    logger.info("No contraindicated drugs found. Trying fallback: drugs requiring adjustment...")
                    
                    # Determine treat vs safety condition
                    treat_indicators = ["pneumonia", "uti", "infection", "hypertension", "hyperlipidemia", "fibrillation", "atrial"]
                    treat_idx = 0
                    for i, cond in enumerate(conditions):
                        cond_lower = cond.lower()
                        if any(indicator in cond_lower for indicator in treat_indicators):
                            treat_idx = i
                            break
                    safety_idx = 1 - treat_idx
                    
                    # Build fallback query for REQUIRES_ADJUSTMENT
                    requested_class = normalized_entities.get('drug_classes', drug_classes)[0].lower() if 'drug_classes' in normalized_entities else drug_classes[0].lower()
                    class_stem = requested_class.rstrip('s')
                    if class_stem == "antibiotic":
                        class_stem = "antibiot"
                        class_filter = f'toLower(d.class) CONTAINS "{class_stem}"'
                    elif class_stem == "anticoagulant" or requested_class == "blood thinner":
                        # Blood thinners use multiple class names: DOAC, Vitamin K Antagonist, LMWH
                        class_filter = '(toLower(d.class) CONTAINS "doac" OR toLower(d.class) CONTAINS "vitamin k antagonist" OR toLower(d.class) CONTAINS "lmwh")'
                    else:
                        class_filter = f'toLower(d.class) CONTAINS "{class_stem}"'
                    
                    fallback_cypher = f"""
MATCH (d:Drug)
WHERE {class_filter}
  AND (d)-[:TREATS]->(:Condition {{name: $treat_condition}})
MATCH (d)-[:REQUIRES_ADJUSTMENT]->(:Condition {{name: $safety_condition}})
RETURN d.name AS drug, d.class AS drug_class, true AS needs_adjustment
ORDER BY d.name
"""
                    
                    try:
                        with self.driver.session(database=self.database) as session:
                            fallback_result = session.run(
                                fallback_cypher,
                                treat_condition=conditions[treat_idx],
                                safety_condition=conditions[safety_idx]
                            )
                            fallback_results = [dict(record) for record in fallback_result]
                            if fallback_results:
                                logger.info(f"Fallback query returned {len(fallback_results)} result(s) requiring adjustment")
                                results = fallback_results
                                # Mark that these are "caution" not "avoid"
                                for r in results:
                                    r['is_caution'] = True
                    except Exception as e:
                        logger.error(f"Fallback query failed: {e}")
                
                # Final validation: If drug class was requested but no results match, return error
                if not results and drug_classes:
                    logger.warning(f"No results found matching drug class '{drug_classes[0]}'")
                    # Return a helpful message instead of falling back to wrong template
                    return {
                        "answer": f"No {drug_classes[0]} medications found that meet the specified constraints in the database. Please consult a healthcare professional for alternative options.",
                        "results": [],
                        "query_type": query_type,
                        "intent": intent,
                        "confidence": 0.0
                    }
                
                if results:
                    logger.info(f"After class compliance check: {len(results)} result(s)")
                    
                    # Determine treat_condition for answer synthesis (use same logic as parameter mapping)
                    if drug_classes and len(conditions) >= 2:
                        # Use the same treat_idx logic we used for parameter mapping
                        treat_indicators = ["pneumonia", "uti", "infection", "hypertension", "hyperlipidemia", "fibrillation", "atrial"]
                        treat_idx = 0
                        for i, cond in enumerate(conditions):
                            cond_lower = cond.lower()
                            if any(indicator in cond_lower for indicator in treat_indicators):
                                treat_idx = i
                                break
                        treat_condition = conditions[treat_idx]
                        safety_condition = conditions[1 - treat_idx] if len(conditions) > 1 else None
                    elif len(conditions) >= 2:
                        treat_condition = conditions[0]
                        safety_condition = conditions[1]
                    elif conditions:
                        # For single condition, check if it's an "avoid" query
                        intent_result = self.detect_query_intent(query, entities)
                        is_avoid_query_for_condition = intent_result["is_avoid_query"]
                        
                        if is_avoid_query_for_condition:
                            # For "avoid" queries: single condition is the safety constraint
                            treat_condition = None  # No treat condition for pure "avoid" queries
                            safety_condition = conditions[0]
                        else:
                            # For "safe" queries: single condition is what we're treating
                            treat_condition = conditions[0]
                            safety_condition = None
                    else:
                        treat_condition = "the condition"
                        safety_condition = None
                    
                    # Use LLM to generate comprehensive answer
                    if len(conditions) >= 2 or drugs:
                        # Filtered list (has constraints)
                        # Detect query intent using LLM (reuse if already detected)
                        if 'is_avoid_query_for_condition' not in locals():
                            intent_result = self.detect_query_intent(query, entities)
                            is_avoid_query = intent_result["is_avoid_query"]
                        else:
                            is_avoid_query = is_avoid_query_for_condition
                        answer = self._synthesize_aggregation_answer(
                            query=query,
                            condition=treat_condition if treat_condition else safety_condition,  # Use safety_condition if no treat_condition
                            results=results,
                            query_type="filtered_list",
                            safety_condition=safety_condition,
                            is_avoid_query=is_avoid_query,
                            execution_context=execution_context
                        )
                    else:
                        # Simple list answer
                        answer = self._synthesize_aggregation_answer(
                            query=query,
                            condition=treat_condition,
                            results=results,
                            query_type="simple_list"
                        )
                    
                    return {
                        "answer": answer,
                        "results": results,
                        "query_type": query_type,
                        "intent": intent,
                        "confidence": 0.90
                    }
                else:
                    logger.warning("LLM-generated Cypher query returned no results")
                    # Log which constraints might have eliminated results
                    if drug_classes:
                        logger.warning(f"  - Drug class filter: {drug_classes[0]}")
                    if conditions:
                        logger.warning(f"  - Condition filter: {conditions}")
                    if drugs:
                        logger.warning(f"  - Drug interaction filter: {drugs[0]}")
                    
            except Exception as e:
                logger.error(f"LLM Cypher generation or execution failed: {e}")
                logger.info("Falling back to pattern-based query generation")
                
                # Fallback to original pattern-based logic
                conditions = entities.get('conditions', [])
                
                if len(conditions) >= 2:
                    treat_condition = conditions[0]
                    safety_condition = conditions[1]
                    
                    cypher = """
                    MATCH (d:Drug)-[:TREATS]->(c:Condition {name: $treat_condition})
                    OPTIONAL MATCH (d)-[contra:CONTRAINDICATED_IN]->(safety:Condition {name: $safety_condition})
                    OPTIONAL MATCH (d)-[adjust:REQUIRES_ADJUSTMENT]->(safety2:Condition {name: $safety_condition})
                    WITH d, contra, adjust
                    WHERE contra IS NULL
                    RETURN d.name AS drug, 
                           d.class AS drug_class,
                           d.pregnancy_category AS pregnancy_category,
                           CASE WHEN adjust IS NOT NULL THEN true ELSE false END AS needs_adjustment
                    ORDER BY d.name
                    """
                    
                    try:
                        with self.driver.session(database=self.database) as session:
                            result = session.run(cypher, 
                                               treat_condition=treat_condition,
                                               safety_condition=safety_condition)
                            results = [dict(record) for record in result]
                        
                        if results:
                            answer = self._synthesize_aggregation_answer(
                                query=query,
                                condition=treat_condition,
                                results=results,
                                query_type="filtered_list",
                                safety_condition=safety_condition
                            )
                            
                            return {
                                "answer": answer,
                                "results": results,
                                "query_type": query_type,
                                "intent": intent,
                                "confidence": 0.85
                            }
                    except Exception as e2:
                        logger.error(f"Fallback Cypher execution failed: {e2}")
        
        elif query_type == "interaction_check":
            # Check if specific drug(s) interact with other drugs or drug classes
            drugs = entities.get('drugs', [])
            drug_classes = entities.get('drug_classes', [])
            
            # Case 2: Check if two specific drugs interact (prioritize this when we have 2 drugs)
            if len(drugs) >= 2:
                drug1, drug2 = drugs[0], drugs[1]
                
                logger.info(f"Checking interaction between '{drug1}' and '{drug2}'")
                
                # Use bidirectional relationship - INTERACTS_WITH can be stored in either direction
                cypher = """
                MATCH (d1:Drug {name: $drug1})-[r:INTERACTS_WITH]-(d2:Drug {name: $drug2})
                RETURN d1.name AS drug1, 
                       d2.name AS drug2,
                       r.severity AS severity,
                       r.description AS description
                """
                
                try:
                    with self.driver.session(database=self.database) as session:
                        logger.debug(f"Executing Cypher: {cypher} with drug1='{drug1}', drug2='{drug2}'")
                        result = session.run(cypher, drug1=drug1, drug2=drug2)
                        records = list(result)
                        logger.info(f"Query returned {len(records)} record(s)")
                        if records:
                            logger.info(f"Found interaction: {records[0]}")
                    
                    if records:
                        interaction = dict(records[0])
                        
                        # Use LLM to generate detailed, patient-friendly explanation
                        answer = self._explain_interaction_with_llm(
                            drug1=interaction['drug1'],
                            drug2=interaction['drug2'],
                            severity=interaction['severity'],
                            description=interaction['description'],
                            query=query
                        )
                        
                        return {
                            "answer": answer,
                            "interaction": interaction,
                            "query_type": query_type,
                            "intent": intent,
                            "confidence": 1.0
                        }
                    else:
                        # Use LLM to generate reassuring but cautious response
                        answer = self._explain_no_interaction_with_llm(drug1, drug2, query)
                        
                        return {
                            "answer": answer,
                            "query_type": query_type,
                            "intent": intent,
                            "confidence": 0.85
                        }
                except Exception as e:
                    logger.error(f"Interaction check failed: {e}")
            
            # Case 1: Check if specific drug interacts with ANY drug in a class
            elif len(drugs) >= 1 and len(drug_classes) >= 1:
                specific_drug = drugs[0]
                drug_class = drug_classes[0]
                
                # Normalize drug class name (remove plurals, standardize)
                drug_class_normalized = drug_class.lower().rstrip('s')  # "antibiotics" -> "antibiotic"
                
                # Map drug class to known class names in database
                class_mapping = {
                    "antibiotic": ["Macrolide", "Penicillin", "Cephalosporin", "Fluoroquinolone", "Tetracycline", "Antibiotic"],
                    "statin": ["Statin"],
                    "blood thinner": ["Anticoagulant"],
                    "ssri": ["SSRI"],
                    "ace inhibitor": ["ACE Inhibitor"],
                }
                
                # Use mapped classes or fallback to contains search
                known_classes = class_mapping.get(drug_class_normalized, [])
                
                if known_classes:
                    # Use IN operator for known drug classes
                    cypher = """
                    MATCH (d1:Drug {name: $drug_name})-[r:INTERACTS_WITH]-(d2:Drug)
                    WHERE d2.class IN $drug_classes
                    RETURN d1.name AS drug1, 
                           d2.name AS drug2,
                           d2.class AS drug_class,
                           r.severity AS severity,
                           r.description AS description
                    ORDER BY r.severity DESC
                    """
                else:
                    # Fallback to CONTAINS search
                    cypher = """
                    MATCH (d1:Drug {name: $drug_name})-[r:INTERACTS_WITH]-(d2:Drug)
                    WHERE toLower(d2.class) CONTAINS toLower($drug_class)
                    RETURN d1.name AS drug1, 
                           d2.name AS drug2,
                           d2.class AS drug_class,
                           r.severity AS severity,
                           r.description AS description
                    ORDER BY r.severity DESC
                    """
                
                try:
                    with self.driver.session(database=self.database) as session:
                        if known_classes:
                            # Pass the list of drug classes
                            result = session.run(cypher, drug_name=specific_drug, drug_classes=known_classes)
                        else:
                            # Pass the drug class string for CONTAINS search
                            result = session.run(cypher, drug_name=specific_drug, drug_class=drug_class)
                        records = [dict(record) for record in result]
                    
                    if records:
                        # Found interactions with drugs in the class
                        interacting_drugs = [r['drug2'] for r in records]
                        
                        # Generate comprehensive answer about class interactions
                        drugs_list = ", ".join(interacting_drugs)
                        severity_levels = [r['severity'] for r in records]
                        max_severity = "MAJOR" if "MAJOR" in severity_levels else "MODERATE"
                        
                        # Build clear evidence string
                        evidence_lines = []
                        for r in records:
                            drug_class_label = r.get('drug_class', drug_class)
                            evidence_lines.append(
                                f"  • {specific_drug} INTERACTS_WITH {r['drug2']} (class: {drug_class_label})\n"
                                f"    Severity: {r['severity']}\n"
                                f"    Description: {r['description']}"
                            )
                        evidence_text = "\n".join(evidence_lines)
                        
                        answer_prompt = get_drug_class_interaction_answer_prompt(
                            query, specific_drug, drug_class, evidence_text, len(records), max_severity
                        )
                        
                        llm_response = llm.invoke(answer_prompt)
                        answer = llm_response.content.strip()
                        
                        return {
                            "answer": answer,
                            "interactions": records,
                            "query_type": query_type,
                            "intent": intent,
                            "confidence": 1.0
                        }
                    else:
                        # No interactions found with drugs in that class
                        answer_prompt = get_no_drug_class_interaction_answer_prompt(query, specific_drug, drug_class)
                        
                        llm_response = llm.invoke(answer_prompt)
                        answer = llm_response.content.strip()
                        
                        return {
                            "answer": answer,
                            "query_type": query_type,
                            "intent": intent,
                            "confidence": 0.85
                        }
                except Exception as e:
                    logger.error(f"Interaction check with drug class failed: {e}")
        
        # If we couldn't handle it with aggregation, return None to fall back to beam search
        return None

    # ------------------------------------------------------------------
    # Get neighbors (safe label)
    # ------------------------------------------------------------------
    def get_neighbors(self, node_name: str, node_type: str) -> List[Dict]:
        label = node_type.split(";")[0]
        cypher = f"""
        MATCH (n:{label} {{name: $name}})-[r]-(neighbor)
        RETURN 
            neighbor.name AS name,
            labels(neighbor)[0] AS type,
            type(r) AS relationship,
            properties(r) AS rel_props
        LIMIT 50
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, name=node_name)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Path scoring
    # ------------------------------------------------------------------
    def score_path(self, path: Path, query: str) -> float:
        score = 100.0

        # Length: prefer shorter but allow multi-hop
        length_penalty = max(0, path.length() - 2) * 4
        score -= length_penalty

        # Query overlap
        query_terms = set(query.lower().split())
        path_terms = set(" ".join(path.nodes).lower().split())
        score += len(query_terms & path_terms) * 12

        # Relationship diversity
        score += len(set(path.relationships)) * 6

        # Medical importance
        rel_bonus = {
            "TREATS": 20,
            "CONTRAINDICATED_IN": 25, "REQUIRES_ADJUSTMENT": 18,
            "INTERACTS_WITH": 20, "CONTRAINDICATES": 20
        }
        for rel in path.relationships:
            score += rel_bonus.get(rel, 5)

        # Complete reasoning (Drug ↔ Condition)
        if len(path.node_types) >= 2 and {"Drug", "Condition"} <= set(path.node_types):
            score += 30

        return max(0, score)

    # ------------------------------------------------------------------
    # Expand paths with LLM guidance
    # ------------------------------------------------------------------
    def expand_paths(self, paths: List[Path], query: str) -> List[Path]:
        new_paths = []

        for path in paths:
            if path.length() >= self.max_depth:
                new_paths.append(path)
                continue

            last_node = path.last_node()
            last_type = path.last_node_type()

            raw_neighbors = self.get_neighbors(last_node, last_type)
            chosen = self.llm_agent.propose_action(
                path_nodes=path.nodes,
                query=query,
                neighbors=raw_neighbors,
            )

            if not chosen:
                chosen = raw_neighbors[: self.beam_width]

            # Track if any valid expansion was made
            expanded = False
            
            for neighbor in chosen:
                if neighbor["name"] in path.nodes:
                    continue

                expanded = True
                justification = neighbor.get("_llm_justification", "Graph expansion")

                new_path = Path(
                    nodes=path.nodes + [neighbor["name"]],
                    node_types=path.node_types + [neighbor["type"]],
                    relationships=path.relationships + [neighbor["relationship"]],
                    evidence=path.evidence + [f"{last_node} {neighbor['relationship']} {neighbor['name']}"],
                    llm_reasoning=path.llm_reasoning + [justification],
                )
                new_path.score = self.score_path(new_path, query)
                new_paths.append(new_path)

            # If no valid expansion was made (dead end), keep the original path
            # This is crucial for beam search - don't lose paths that hit dead ends
            if not expanded:
                new_paths.append(path)

        return new_paths

    # ------------------------------------------------------------------
    # Main beam search
    # ------------------------------------------------------------------
    def beam_search(self, query: str, starting_nodes: List[Tuple[str, str]] = None) -> List[Path]:
        """
        Perform beam search to find reasoning paths.
        
        Args:
            query: User's natural language query
            starting_nodes: Optional pre-computed starting nodes. If None, will find them.
        
        Returns:
            List of Path objects sorted by score
        """
        logger.info(f"Starting beam search: {query}")

        # Use provided starting nodes or find them
        if starting_nodes is None:
            starting_nodes = self.find_starting_nodes(query)
        
        if not starting_nodes:
            logger.warning("No starting nodes")
            return []
        
        starting = starting_nodes

        # Initialize
        current_paths = []
        for name, ntype in starting:
            path = Path(nodes=[name], node_types=[ntype], relationships=[])
            path.score = self.score_path(path, query)
            current_paths.append(path)

        # Iterate
        for depth in range(self.max_depth):
            logger.info(f"Depth {depth + 1}/{self.max_depth} | Paths: {len(current_paths)}")
            expanded = self.expand_paths(current_paths, query)
            expanded.sort(key=lambda p: p.score, reverse=True)
            current_paths = expanded[: self.beam_width]
        # After beam search loop
        if not current_paths:
            logger.warning("No valid paths after beam search")
            return []
        
        logger.info(f"Beam search complete. Best score: {current_paths[0].score:.1f}")
        return current_paths

    # ------------------------------------------------------------------
    # Answer generation (LLM-enhanced)
    # ------------------------------------------------------------------
    def _generate_answer_with_llm(self, path: Path, query: str, all_paths: List[Path] = None, starting_nodes: List[Tuple[str, str]] = None) -> str:
        """Use LLM to generate natural language answer from reasoning path(s)"""

        # Format raw paths simply - let LLM decide what to show
        paths_data = []
        if all_paths:
            for i, p in enumerate(all_paths[:5]):  # Top 5 paths
                path_info = {
                    "path_number": i + 1,
                    "score": p.score,
                    "nodes": p.nodes,
                    "node_types": p.node_types,
                    "relationships": p.relationships,
                    "path_string": " → ".join(p.nodes)
                }
                paths_data.append(path_info)

        # Extract all unique Drug nodes for reference
        all_drug_nodes = set()
        for p in all_paths or [path]:
            for i, node in enumerate(p.nodes):
                if i < len(p.node_types) and p.node_types[i] == "Drug":
                    all_drug_nodes.add(node)

        # Simple formatted evidence - LLM will decide what to show
        evidence = f"""Beam Search Results:
- Total paths found: {len(all_paths) if all_paths else 1}
- Unique drugs found: {', '.join(sorted(all_drug_nodes)) if all_drug_nodes else 'None'}

Paths (sorted by relevance score):
"""
        for p_info in paths_data:
            evidence += f"""
Path {p_info['path_number']} (score: {p_info['score']:.1f}):
  Nodes: {' → '.join(p_info['nodes'])}
  Relationships: {' → '.join(p_info['relationships'])}
"""

        prompt = get_path_based_answer_prompt(query, evidence)

        try:
            response = llm.invoke(prompt)
            answer = response.content.strip()
            logger.info("Generated answer using LLM")
            return answer
        except Exception as e:
            logger.warning(f"LLM answer generation failed: {e}, using fallback")
            return evidence

    def _generate_answer(self, path: Path, query: str) -> str:
        """Legacy template-based answer generation (fallback)"""
        if path.length() <= 1:
            return f"Found: {path.nodes[0]}."

        rel_map = {
            "TREATS": "treats",
            "CONTRAINDICATED_IN": "is contraindicated in",
            "REQUIRES_ADJUSTMENT": "requires dose adjustment in",
            "INTERACTS_WITH": "interacts with",
            "CONTRAINDICATES": "contraindicates",
        }

        steps = []
        for i in range(len(path.relationships)):
            rel_text = rel_map.get(path.relationships[i], path.relationships[i].lower())
            steps.append(f"{path.nodes[i]} {rel_text} {path.nodes[i+1]}")

        return " → ".join(steps) + "."
    
    def _generate_cannot_answer_message(self, query: str, query_type: str) -> str:
        """Use LLM to generate helpful message when query cannot be answered"""
        
        # Get available schema to explain what IS in the graph
        schema_info = ""
        if self.query_planner and self.query_planner.schema_inspector:
            try:
                schema = self.query_planner.schema_inspector.get_schema()
                schema_summary = schema.to_summary()
                schema_info = f"\n\nAvailable in the knowledge graph:\n{schema_summary}"
            except:
                schema_info = "\n\nThe graph contains: Drug and Condition nodes with relationships like TREATS, CONTRAINDICATED_IN, INTERACTS_WITH, and REQUIRES_ADJUSTMENT."
        
        prompt = get_cannot_answer_message_prompt(query, query_type, schema_info)

        try:
            response = llm.invoke(prompt)
            answer = response.content.strip()
            logger.info("Generated 'cannot answer' explanation using LLM")
            return answer
        except Exception as e:
            logger.warning(f"LLM explanation failed: {e}, using default message")
            return f"I cannot answer this question because the current knowledge graph doesn't contain the necessary information. The graph currently models drug-condition relationships (TREATS, CONTRAINDICATED_IN, INTERACTS_WITH), but doesn't include drug similarity, class hierarchies, or cross-allergenicity data that would be needed to answer your question."

    def answer_question(self, query: str) -> Dict[str, Any]:
        """Main entry point - uses beam search for all queries"""
        
        # Classify query type for informational purposes
        query_type = self.classify_query_type(query)
        logger.info(f"Query classified as: {query_type} (routing to beam search)")
        
        # Use beam search for all queries
        logger.info("Using beam search for query")
        
        # Find starting nodes once (not twice!)
        starting_nodes = self.find_starting_nodes(query)
        
        if not starting_nodes:
            # Use LLM to generate helpful explanation of why this can't be answered
            explanation = self._generate_cannot_answer_message(query, query_type)
            return {
                "answer": explanation,
                "paths": [],
                "query_type": query_type,
                "confidence": 0.0,
            }
        
        # Pass starting nodes to beam search (avoids re-calling find_starting_nodes)
        paths = self.beam_search(query, starting_nodes=starting_nodes)
        
        if not paths:
            # Use LLM to generate helpful explanation of why this can't be answered
            explanation = self._generate_cannot_answer_message(query, query_type)
            return {
                "answer": explanation,
                "paths": [],
                "query_type": query_type,
                "confidence": 0.0,
            }

        top = paths[0]
        # Use LLM-enhanced answer generation for better natural language responses
        answer = self._generate_answer_with_llm(top, query, all_paths=paths, starting_nodes=starting_nodes)
        confidence = min(1.0, top.score / 100.0)

        return {
            "answer": answer,
            "paths": [p.to_dict() for p in paths[:3]],
            "query_type": query_type,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        self.driver.close()


# ----------------------------------------------------------------------
# CLI for testing
# ----------------------------------------------------------------------
def main():
    reasoner = BeamSearchReasoner(beam_width=3, max_depth=4)
    print("\nType 'quit' to exit.\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        print()  # Empty line for readability
        result = reasoner.answer_question(query)
        
        # Show query type classification
        query_type = result.get('query_type', 'unknown')
        print(f"[Query Type: {query_type}]")
        print()
        
        # Show answer
        print(result['answer'])
        print(f"\nConfidence: {result['confidence']:.1%}")

        # Show paths if available (for beam search results)
        if result.get('paths'):
            print("\nReasoning Paths:")
            for i, p in enumerate(result["paths"][:3], 1):
                print(f"  Path {i} (score: {p['score']:.1f}):")
                print(f"    {' → '.join(p['nodes'])}")
                if p.get("llm_reasoning") and p['llm_reasoning']:
                    print(f"    LLM: {p['llm_reasoning'][-1]}")
        
        print("\n" + "="*70 + "\n")

    reasoner.close()


if __name__ == "__main__":
    main()