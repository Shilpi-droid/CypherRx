"""
Prompts and Domain Knowledge for Medical Query Planning
Contains:
- Entity mappings (colloquial → formal medical terms)
- Few-shot examples for Cypher generation
- System prompts for LLM guidance
"""

from typing import List, Dict, Any

# ==============================================================================
# Entity Normalization Mappings
# ==============================================================================

ENTITY_MAPPINGS = {
    # Conditions - must match database exactly
    "irregular heartbeat": "Atrial Fibrillation",
    "irregular heart beat": "Atrial Fibrillation",
    "a-fib": "Atrial Fibrillation",
    "afib": "Atrial Fibrillation",
    "atrial fib": "Atrial Fibrillation",
    "atrial fibrillation": "Atrial Fibrillation",
    "high blood pressure": "Hypertension",
    "blood pressure": "Hypertension",
    "hbp": "Hypertension",
    "high cholesterol": "Hyperlipidemia",
    "cholesterol": "Hyperlipidemia",
    "diabetes": "Type 2 Diabetes Mellitus",
    "type 2 diabetes": "Type 2 Diabetes Mellitus",
    "diabetic": "Type 2 Diabetes Mellitus",
    "kidney disease": "Chronic Kidney Disease",
    "kidney problems": "Chronic Kidney Disease",
    "renal disease": "Chronic Kidney Disease",
    "ckd": "Chronic Kidney Disease",
    "pneumonia": "Community-Acquired Pneumonia",
    "lung infection": "Community-Acquired Pneumonia",
    "uti": "Urinary Tract Infection",
    "urinary tract infection": "Urinary Tract Infection",
    "bladder infection": "Urinary Tract Infection",
    "depression": "Major Depressive Disorder",
    "depressed": "Major Depressive Disorder",
    "blood clots": "Deep Vein Thrombosis",
    "blood clot": "Deep Vein Thrombosis",
    "dvt": "Deep Vein Thrombosis",
    "thrombosis": "Deep Vein Thrombosis",
    "pregnant": "Pregnancy",
    "pregnancy": "Pregnancy",
    "expecting": "Pregnancy",
    "strep throat": "Streptococcal infection",
    "streptococcal pharyngitis": "Streptococcal infection",
    "asthma": "Asthma",
    "copd": "COPD",
    "gout": "Gout",
    "bph": "Benign Prostatic Hyperplasia",
    "enlarged prostate": "Benign Prostatic Hyperplasia",
    "erectile dysfunction": "Erectile Dysfunction",
    # "ed": "Erectile Dysfunction",  # Removed - causes false matches in words like "need", "used", "treated"
    
    # Drugs (common ones)
    "birth control": "Contraceptive",
    "contraceptive": "Contraceptive",
    "oral contraceptive": "Contraceptive",
    "blood thinners": "Anticoagulant",
    "blood thinner": "Anticoagulant",
    "anticoagulants": "Anticoagulant",
    "anticoagulant": "Anticoagulant",
    "antidepressants": "SSRI",
    "antidepressant": "SSRI",
    "ssri": "SSRI",
    "aspirin": "Aspirin",
    "warfarin": "Warfarin",
    "metformin": "Metformin",
    "insulin": "Insulin Glargine",  # Default insulin reference
    "insulin glargine": "Insulin Glargine",
    "lisinopril": "Lisinopril",
    "atorvastatin": "Atorvastatin",
    "amoxicillin": "Amoxicillin",
    
    # Side effects / Symptoms
    "dizziness": "Dizziness",
    "dizzy": "Dizziness",
    "fatigue": "Fatigue",
    "tired": "Fatigue",
    "tiredness": "Fatigue",
    
    # Drug classes
    "antibiotics": "Antibiotic",
    "antibiotic": "Antibiotic",

    "ibuprofen": "Ibuprofen",
    "advil": "Ibuprofen",
    "motrin": "Ibuprofen",
    "nsaid": "NSAID",
    "painkiller": "NSAID",
}


# ==============================================================================
# Default Schema (Fallback)
# ==============================================================================

DEFAULT_SCHEMA = """
**Knowledge Graph Schema** (Default - may not be accurate):
- Nodes: Drug, Condition, Effect, Enzyme, Ingredient
- Relationships: TREATS, CONTRAINDICATED_IN, INTERACTS_WITH, REQUIRES_ADJUSTMENT, CONTRAINDICATES
"""


# ==============================================================================
# Drug Class Information
# ==============================================================================

DRUG_CLASS_INFO = """
**DRUG PROPERTIES**:
- Every Drug node has a `class` property containing the drug class name
- Drug classes include: "Statin", "Macrolide", "Penicillin", "Cephalosporin", "Fluoroquinolone", "Tetracycline", "SSRI", "ACE Inhibitor", etc.
- **Antibiotic classes**: When user says "antibiotic", they mean drugs with class in: ["Penicillin", "Cephalosporin", "Macrolide", "Fluoroquinolone", "Tetracycline", "Nitrofuran", "Antifolate", "Sulfonamide", "Glycopeptide", "Nitroimidazole"]

**FILTERING BY DRUG CLASS**:
To find drugs of a specific class (e.g., antibiotics) that interact with a drug:
MATCH (d:Drug {{name: "Simvastatin"}})-[:INTERACTS_WITH]-(a:Drug)
WHERE a.class IN ["Macrolide", "Penicillin", "Cephalosporin", "Fluoroquinolone", "Tetracycline"]
RETURN a.name AS name, "Drug" AS type

Or use CONTAINS for partial matching:
WHERE toLower(a.class) CONTAINS "macrolide"
"""


# ==============================================================================
# Entity Naming Conventions
# ==============================================================================

ENTITY_NAMING_CONVENTION = """
**Entity Naming Convention**:
- Use EXACT formal medical terminology (not colloquial names)
- Drugs: "Warfarin", "Aspirin", "Metformin", "Insulin Glargine", "Lisinopril", "Atorvastatin", "Amoxicillin", "Azithromycin"
- Conditions: "Hypertension", "Type 2 Diabetes Mellitus", "Atrial Fibrillation", "Chronic Kidney Disease", "Pregnancy", "Community-Acquired Pneumonia"
- Common mappings:
  * "insulin" → "Insulin Glargine"
  * "diabetes" → "Type 2 Diabetes Mellitus"
  * "irregular heartbeat" → "Atrial Fibrillation"
  * "kidney disease" → "Chronic Kidney Disease"
"""


# ==============================================================================
# Query Strategy Guidelines
# ==============================================================================

QUERY_STRATEGY = """
**Query Strategy**:
1. For "Which drugs treat X?": Start from the condition node
2. For "What does drug X interact with?": Start from the drug node
3. For "Which X are safe in Y?": Start from drugs treating X, filter by contraindications/interactions
"""


# ==============================================================================
# Important Cypher Rules
# ==============================================================================

CYPHER_RULES = """
**Important Rules**:
- ALWAYS use exact property matching: {{name: "Exact Name"}}
- For filtering (NOT conditions), use WHERE NOT (pattern)
- MUST return: name AS name, type AS type
- Keep queries simple and focused on finding starting nodes
- **CRITICAL**: type() function ONLY works on relationships, NOT nodes
  * For nodes: Use labels(node)[0] or a literal string like "Drug"
  * For relationships: Use type(relationship)
  * WRONG: type(n) where n is a node
  * RIGHT: labels(n)[0] or "Drug" for nodes

**CRITICAL: INTERACTS_WITH RELATIONSHIP DIRECTION**:
- INTERACTS_WITH is BIDIRECTIONAL - relationships can be stored in either direction
- ALWAYS use undirected relationship pattern: (d1)-[:INTERACTS_WITH]-(d2)
- NEVER use directed: (d1)-[:INTERACTS_WITH]->(d2) or <-[:INTERACTS_WITH]-(d2)
- This ensures you find interactions regardless of how they're stored in the graph
- Example: MATCH (d1:Drug {name: "Warfarin"})-[r:INTERACTS_WITH]-(d2:Drug {name: "Ibuprofen"})

**CRITICAL SAFETY DEFINITION**:
- In this knowledge graph, "safe" means: **NO `INTERACTS_WITH` relationship**
- If two drugs have NO `INTERACTS_WITH` edge → they are **safe together**
- If there IS an `INTERACTS_WITH` edge → **not safe**
- Use bidirectional check: NOT (d)-[:INTERACTS_WITH]-(:Drug {name: "Warfarin"})
"""


# ==============================================================================
# Few-Shot Examples
# ==============================================================================
FEW_SHOT_EXAMPLES = """
**CRITICAL: RELATIONSHIP DIRECTION**
- TREATS goes FROM Drug TO Condition
- ALWAYS use: (d:Drug)-[:TREATS]->(c:Condition)
- NEVER use: (c)<-[:TREATS]-(d)
- TREATED_BY does NOT exist

**Examples**:

User: "Which drugs treat Hypertension and are safe in Pregnancy?"
→ MATCH (d:Drug)-[:TREATS]->(c:Condition {name: "Hypertension"})
   WHERE NOT (d)-[:CONTRAINDICATED_IN]->(:Condition {name: "Pregnancy"})
   RETURN d.name AS name, "Drug" AS type

User: "What does Warfarin interact with?"
→ MATCH (d:Drug {name: "Warfarin"})-[:INTERACTS_WITH]-(other)
   RETURN other.name AS name, labels(other)[0] AS type

User: "Does Simvastatin interact with any antibiotics?"
→ MATCH (d:Drug {name: "Simvastatin"})-[:INTERACTS_WITH]-(a:Drug)
   WHERE toLower(a.class) CONTAINS "antibiotic"
   RETURN a.name AS name, "Drug" AS type

User: "Alternative Antibiotic for Streptococcal infection that don't interact with Contraceptive?"
→ MATCH (d:Drug)-[:TREATS]->(c:Condition {name: "Streptococcal infection"})
   WHERE toLower(d.class) CONTAINS "antibiotic"
     AND NOT (d)-[:INTERACTS_WITH]-(:Drug {name: "Contraceptive"})
   RETURN d.name AS name, "Drug" AS type

User: "What drugs treat Diabetes?"
→ MATCH (d:Drug)-[:TREATS]->(c:Condition {name: "Type 2 Diabetes Mellitus"})
   RETURN d.name AS name, "Drug" AS type

User: "What drugs treat UTI?"
→ MATCH (d:Drug)-[:TREATS]->(c:Condition {name: "Urinary Tract Infection"})
   RETURN d.name AS name, "Drug" AS type

User: "Safe antibiotic for UTI with Warfarin?"
→ MATCH (d:Drug)-[:TREATS]->(c:Condition {name: "Urinary Tract Infection"})
   WHERE (toLower(d.class) IN ["penicillin", "cephalosporin", "macrolide", "fluoroquinolone", "tetracycline", "nitrofuran", "antifolate", "sulfonamide", "glycopeptide", "nitroimidazole"])
     AND NOT (d)-[:INTERACTS_WITH]-(:Drug {name: "Warfarin"})
   RETURN d.name AS name, "Drug" AS type

User: "What condition connects Metformin and Insulin Glargine?"
→ MATCH (d1:Drug {name: "Metformin"})-[:TREATS]->(c:Condition)<-[:TREATS]-(d2:Drug {name: "Insulin Glargine"})
   RETURN c.name AS name, labels(c)[0] AS type
"""
# ==============================================================================
# System Prompt Template
# ==============================================================================

def get_query_planner_system_prompt(schema_info: str) -> str:
    """
    Generate the system prompt for query planning
    
    Args:
        schema_info: Database schema information (from inspector or default)
        
    Returns:
        Complete system prompt string
    """
    return f"""
You are a medical reasoning planner for a knowledge graph database.

**Goal**: Generate a Cypher query to find the best starting nodes for answering the user's medical question.

{schema_info}

{DRUG_CLASS_INFO}

{ENTITY_NAMING_CONVENTION}

{QUERY_STRATEGY}

{CYPHER_RULES}

{FEW_SHOT_EXAMPLES}

**Now generate the Cypher query for the user's question. Return ONLY the Cypher query, no explanations.**
"""


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_user_prompt(normalized_query: str, entities: dict = None) -> str:
    """
    Generate user prompt for query planning with entity context
    
    Args:
        normalized_query: The normalized user query
        entities: Extracted entities (drugs, conditions, drug_classes)
        
    Returns:
        Formatted user prompt with entity information
    """
    if entities is None:
        entities = {"drugs": [], "conditions": [], "drug_classes": []}
    
    # Build entity context
    entity_info = ""
    if any(entities.values()):
        entity_parts = []
        if entities.get("drugs"):
            entity_parts.append(f"Drugs: {', '.join(entities['drugs'])}")
        if entities.get("conditions"):
            entity_parts.append(f"Conditions: {', '.join(entities['conditions'])}")
        if entities.get("drug_classes"):
            entity_parts.append(f"Drug classes: {', '.join(entities['drug_classes'])}")
        
        if entity_parts:
            entity_info = f"\n\nExtracted entities:\n" + "\n".join(f"- {part}" for part in entity_parts)
    
    return f'User query: "{normalized_query}"{entity_info}'


def get_correction_prompt(issues: list, suggestions: list, original_query: str) -> str:
    """Generate prompt for query correction"""
    issues_text = "\n".join(issues)
    suggestions_text = "\n".join(suggestions)
    
    return f"""
The previous Cypher query had issues:
{issues_text}

Suggestions:
{suggestions_text}

Please regenerate the Cypher query using the correct schema elements from the ACTUAL DATABASE SCHEMA provided earlier.
Original query: "{original_query}"
"""


# ==============================================================================
# Beam Search Prompts
# ==============================================================================

# Query Classification Few-Shot Examples
QUERY_CLASSIFICATION_FEW_SHOT_EXAMPLES = """
Here are examples of correct classifications:

Example 1:
Query: "What medications treat diabetes?"
Classification: simple_aggregation
Reasoning: Asking for a list of medications without constraints

Example 2:
Query: "What treats irregular heartbeat?"
Classification: simple_aggregation
Reasoning: Requesting list of treatments for a single condition

Example 3:
Query: "Can I take Warfarin with Azithromycin?"
Classification: interaction_check
Reasoning: Asking if two specific drugs can be taken together - this is an interaction check

Example 3b:
Query: "Does Warfarin interact with Azithromycin?"
Classification: interaction_check
Reasoning: Direct question about drug-drug interaction

Example 4:
Query: "What blood pressure medications are safe for pregnancy?"
Classification: filtered_aggregation
Reasoning: Requesting list with multiple constraints (treats BP AND safe for pregnancy)

Example 5:
Query: "Which antibiotics can treat urinary tract infections?"
Classification: simple_aggregation
Reasoning: Requesting list of specific drug class for a condition

Example 6:
Query: "What condition connects Metformin and Insulin?"
Classification: path_traversal
Reasoning: Asking about relationship/connection between two entities

Example 7:
Query: "What medications can treat irregular heartbeat (Atrial Fibrillation)?"
Classification: simple_aggregation
Reasoning: Requesting list of medications for a condition, no constraints

Example 8:
Query: "Does Simvastatin interact with any antibiotics?"
Classification: interaction_check
Reasoning: Asking about drug interactions with specific drug

Example 9:
Query: "What diabetes drugs are safe for someone with kidney disease?"
Classification: filtered_aggregation
Reasoning: Multiple constraints - treats diabetes AND safe for kidney disease

Example 10:
Example:
Query: "If Apixaban causes problems with my kidneys, what similar blood thinner can I use?"
Classification: filtered_aggregation
Reasoning: Seeking alternative anticoagulant with same indication but better renal safety profile.

"""


def get_query_classification_prompt(query: str) -> str:
    """Generate prompt for query type classification"""
    return f"""{QUERY_CLASSIFICATION_FEW_SHOT_EXAMPLES}

Now classify this query and return ONLY valid JSON in this exact format:

{{
  "query_type": "interaction_check",
  "reasoning": "brief explanation"
}}

Query: "{query}"

Valid query_type values:
- "interaction_check": Asking if two specific drugs interact or can be taken together
- "filtered_aggregation": Requesting a list with multiple constraints (AND, safe, without, avoid)
- "simple_aggregation": Requesting a list of items (drugs, conditions) for a single purpose
- "path_traversal": Asking about relationships or connections between entities

Return ONLY the JSON object, no other text:"""


def get_entity_normalization_prompt(entity: str, entity_type: str, database_entities: List[str]) -> str:
    """Generate prompt for entity normalization with LLM"""
    return f"""Match the user's term to a database term. Return ONLY the matched term, nothing else.

User's term: "{entity}"
Type: {entity_type}

Database options:
{chr(10).join(f"- {opt}" for opt in database_entities[:30])}

Rules:
1. Find the best matching term from the database list
2. Return ONLY the exact term from the list above
3. If no good match exists, return exactly: NO_MATCH
4. Do not add explanations, quotes, or extra text

Your answer (one term only):"""


def get_entity_extraction_prompt(query: str) -> str:
    """Generate prompt for entity extraction with LLM"""
    return f"""You are a medical entity extractor. Extract ALL drugs, conditions, and drug classes from the query.

Query: "{query}"

Rules:
- Drugs: Warfarin, Simvastatin, Amoxicillin, Clarithromycin, etc. (SPECIFIC medication names)
- Drug classes: antibiotics, statins, blood thinners, etc. (CATEGORIES of drugs)
- Conditions: diabetes, hypertension, allergy, etc.

CRITICAL DISTINCTION:
- "Simvastatin" = specific drug → put in "drugs"
- "antibiotics" or "any antibiotics" = drug class → put in "drug_classes"
- "Amoxicillin" = specific antibiotic → put in "drugs"

Examples:
Query: "Does Simvastatin interact with any antibiotics?"
{{
  "drugs": ["Simvastatin"],
  "conditions": [],
  "drug_classes": ["antibiotics"]
}}

Query: "Can I take Warfarin with Azithromycin?"
{{
  "drugs": ["Warfarin", "Azithromycin"],
  "conditions": [],
  "drug_classes": []
}}

Query: "What blood thinners are safe for pregnancy?"
{{
  "drugs": [],
  "conditions": ["pregnancy"],
  "drug_classes": ["blood thinners"]
}}

Return ONLY valid JSON:
{{"drugs": [...], "conditions": [...], "drug_classes": [...]}}"""


def get_interaction_explanation_prompt(drug1: str, drug2: str, severity: str, description: str, query: str) -> str:
    """Generate prompt for explaining drug interactions"""
    return f"""EDUCATIONAL CONTEXT: This is a student AI/ML project demonstrating knowledge graphs.
This is NOT giving medical advice to real patients. Provide factual information from the graph.

Explain this drug interaction found in the educational knowledge graph:

User Question: "{query}"

Interaction Details from Graph:
- Drug 1: {drug1}
- Drug 2: {drug2}
- Severity: {severity}
- Technical Description: {description}

Instructions:
1. Start with a clear statement about the interaction severity
2. Explain WHAT happens when these drugs interact (in simple terms)
3. Explain WHY this is concerning from a pharmacological perspective
4. Describe what would typically be monitored or considered clinically
5. Use emojis appropriately (🔴 for MAJOR, 🟡 for MODERATE)
6. Keep it informative and educational

Generate a clear, educational explanation:"""


def get_no_interaction_explanation_prompt(drug1: str, drug2: str, query: str) -> str:
    """Generate prompt for explaining when no interaction is found"""
    return f"""EDUCATIONAL CONTEXT: This is a student AI/ML project demonstrating knowledge graphs.
This is analyzing an educational database. Provide factual information about what was found.

User Question: "{query}"

Situation: No documented interaction found between {drug1} and {drug2} in the knowledge graph database.

Instructions:
1. State that no interaction was found in the database
2. Explain what this means based on the graph data
3. Note that this database is for educational/demonstration purposes
4. Keep it clear and informative
5. Keep it friendly and factual

Generate the factual response:"""


def get_simple_aggregation_answer_prompt(query: str, condition: str, drugs_text: str) -> str:
    """Generate prompt for synthesizing simple aggregation answers"""
    return f"""EDUCATIONAL CONTEXT: This is a student AI/ML project demonstrating knowledge graphs.
This is analyzing an educational medical database. Provide factual information from the graph.

User Question: "{query}"

Medications found in the knowledge graph that treat {condition}:
{drugs_text}

Instructions:
1. Start with a brief introductory sentence about what was found in the graph
2. List the medications organized by drug class
3. Keep it concise but informative
4. Use a clear, educational tone

Generate the factual answer:"""


def get_filtered_aggregation_answer_prompt(query: str, condition: str, safety_condition: str, 
                                         safe_text: str, adjust_text: str, is_avoid_query: bool = False,
                                         execution_context: Dict[str, Any] = None) -> str:
    """Generate prompt for synthesizing filtered aggregation answers
    
    Args:
        query: Original user query
        condition: Condition being treated (or safety condition for avoid queries)
        safety_condition: Safety constraint condition
        safe_text: Text for safe/avoid drugs (depends on is_avoid_query)
        adjust_text: Text for drugs requiring adjustment
        is_avoid_query: Whether this is an "avoid" query
        execution_context: Context from earlier execution steps
    """
    
    if is_avoid_query:
        # For "avoid" queries: show contraindicated drugs (to avoid) and drugs requiring adjustment (caution)
        
        # Build context section from execution_context
        context_section = ""
        if execution_context:
            context_section = "\n" + "="*80 + "\n"
            context_section += "EXECUTION CONTEXT - WHAT HAPPENED IN EARLIER STEPS:\n"
            context_section += "="*80 + "\n"
            if execution_context.get("query_intent"):
                qi = execution_context["query_intent"]
                context_section += f"1. Query Intent Detected:\n"
                context_section += f"   - is_avoid_query = {qi.get('is_avoid_query')}\n"
                context_section += f"   - intent_type = {qi.get('intent_type')}\n"
                context_section += f"   - reasoning = {qi.get('reasoning')}\n\n"
            if execution_context.get("cypher_query"):
                context_section += f"2. Cypher Query That Was Executed:\n"
                context_section += f"   {execution_context['cypher_query']}\n\n"
                context_section += f"   ⚠️ CRITICAL: This query uses MATCH (d)-[:CONTRAINDICATED_IN]->\n"
                context_section += f"   This means it searched for drugs that are CONTRAINDICATED (prohibited/unsafe).\n\n"
            if execution_context.get("cypher_parameters"):
                context_section += f"3. Parameters Used:\n"
                context_section += f"   {execution_context['cypher_parameters']}\n\n"
            context_section += "="*80 + "\n"
            context_section += "CONCLUSION FROM EXECUTION CONTEXT:\n"
            context_section += "="*80 + "\n"
            context_section += "The drugs listed below were found because the Cypher query searched for\n"
            context_section += "CONTRAINDICATED_IN relationships. This means these drugs are:\n"
            context_section += "- CONTRAINDICATED (prohibited/unsafe)\n"
            context_section += "- NOT safe options\n"
            context_section += "- NOT 'may require caution'\n"
            context_section += "- MUST be avoided\n"
            context_section += "="*80 + "\n\n"
        
        return f"""EDUCATIONAL CONTEXT: This is a student AI/ML project demonstrating knowledge graphs.
This is analyzing an educational medical database. Provide factual information from the graph.

User Question: "{query}"
{context_section}

================================================================================
CRITICAL INFORMATION - READ THIS FIRST:
================================================================================
The user is asking which medications to AVOID. 

The medications listed below were found by executing a Cypher query that searched for drugs with the relationship CONTRAINDICATED_IN. This means these drugs are CONTRAINDICATED (prohibited/unsafe) for {safety_condition if safety_condition else condition}.

These drugs are NOT safe options. They are NOT "may require caution". They are CONTRAINDICATED and MUST be avoided.

================================================================================

❌ Medications to AVOID (contraindicated in graph - DO NOT USE):
{safe_text}

⚠️ Medications requiring dose adjustment (use with caution, per graph data):
{adjust_text}

================================================================================
MANDATORY ANSWER FORMAT - FOLLOW EXACTLY:
================================================================================

1. START your answer with this EXACT sentence:
   "The following medications are CONTRAINDICATED and should be AVOIDED for {safety_condition if safety_condition else condition}:"

2. For EACH medication in the "Medications to AVOID" list, use this format:
   "[Drug Name] is CONTRAINDICATED and should be AVOIDED for {safety_condition if safety_condition else condition} because [reason]."

3. DO NOT use these phrases:
   - "may require caution"
   - "should consult their healthcare provider to determine the safest option"
   - "generally considered safer"
   - "use with caution"
   - "assess suitability"
   
   These phrases imply the drugs might be safe. They are NOT safe - they are CONTRAINDICATED.

4. USE these phrases instead:
   - "is CONTRAINDICATED"
   - "should be AVOIDED"
   - "must not be used"
   - "do not use"
   - "is not suitable"

5. Explain WHY each medication is contraindicated (e.g., renal clearance issues, risk of bleeding, etc.)

6. If there are medications requiring adjustment, clearly separate them as "use with caution" (not contraindicated)

================================================================================

Generate your answer following the MANDATORY FORMAT above. Start with the exact sentence from point 1."""
    else:
        # For "safe" queries: show safe drugs and drugs requiring adjustment
        return f"""EDUCATIONAL CONTEXT: This is a student AI/ML project demonstrating knowledge graphs.
This is analyzing an educational medical database. Provide factual information from the graph.

User Question: "{query}"

Context: Analyzing medications in the graph that treat {condition} and their safety profile for {safety_condition}

✅ Safe medications (no contraindication found in graph):
{safe_text}

⚠️ Medications requiring dose adjustment (per graph data):
{adjust_text}

Instructions:
1. Clearly state which medications are safe based on the graph data
2. Mention which need dose adjustments (if any) according to the graph
3. Explain the pharmacological reasoning for these constraints
4. Keep it clear, factual, and educational

Generate a comprehensive, factual answer:"""


def get_query_intent_detection_prompt(query: str, entities: Dict[str, List[str]]) -> str:
    """Generate prompt for detecting if query is asking for 'avoid' or 'safe' drugs"""
    conditions = entities.get('conditions', [])
    drug_classes = entities.get('drug_classes', [])
    
    return f"""Analyze this medical query to determine if it's asking for drugs to AVOID or drugs that are SAFE.

Query: "{query}"

Extracted entities:
- Conditions: {conditions}
- Drug classes: {drug_classes}

Determine the query intent:
- If the query asks "which should I avoid?", "which is unsafe?", "which should not be taken?", "which is contraindicated?" → is_avoid_query = true, intent_type = "avoid"
- If the query asks "which is safe?", "which can I take?", "which is recommended?" → is_avoid_query = false, intent_type = "safe"
- If the query just asks "which [drugs] treat [condition]?" without safety/avoidance language → is_avoid_query = false, intent_type = "neutral"

Examples:
- "Which blood thinners should I avoid?" → is_avoid_query = true, intent_type = "avoid"
- "Which blood thinners are safe for pregnancy?" → is_avoid_query = false, intent_type = "safe"
- "Which antibiotics treat pneumonia?" → is_avoid_query = false, intent_type = "neutral"
- "I have CKD. Which anticoagulants are unsafe?" → is_avoid_query = true, intent_type = "avoid"

Analyze the query and return the intent."""


def get_intent_analysis_prompt(query: str, entities: Dict[str, List[str]]) -> str:
    """Generate prompt for analyzing query intent"""
    return f"""Analyze this medical query and describe what information is being requested.

Query: "{query}"

Extracted entities:
- Drugs: {entities.get('drugs', [])}
- Conditions: {entities.get('conditions', [])}
- Drug classes: {entities.get('drug_classes', [])}

Describe:
1. What is the PRIMARY goal? (e.g., "find drugs that treat X", "check drug interaction", "find safe alternatives")
2. What CONSTRAINTS exist? (e.g., "safe for pregnancy", "no kidney problems", "don't interact with Y")
3. What should be FILTERED OUT? (e.g., "contraindicated drugs", "drugs that interact")

Be concise and specific."""


def get_cypher_generation_prompt(query: str, intent: str, entities: Dict[str, List[str]], 
                                 schema_info: str = "") -> str:
    """Generate prompt for LLM to create Cypher query based on intent"""
    
    drugs = entities.get('drugs', [])
    conditions = entities.get('conditions', [])
    drug_classes = entities.get('drug_classes', [])
    
    return f"""You are a Cypher query generator for a medical knowledge graph.

Query: "{query}"

Intent Analysis:
{intent}

Extracted Entities:
- Drugs: {drugs}
- Conditions: {conditions}
- Drug classes: {drug_classes}

{schema_info}

**CRITICAL RULES**:
1. INTERACTS_WITH is BIDIRECTIONAL - always use: (d1)-[:INTERACTS_WITH]-(d2)
2. TREATS goes FROM Drug TO Condition: (d:Drug)-[:TREATS]->(c:Condition)
3. Use exact property matching: {{name: "Exact Name"}}
4. For drug class filtering, use the STEM form directly in the query (not as parameter):
   - "antibiotics" → use: WHERE toLower(d.class) CONTAINS "antibiot"
   - "statins" → use: WHERE toLower(d.class) CONTAINS "statin"
   - "blood thinners" → use: WHERE (toLower(d.class) CONTAINS "doac" OR toLower(d.class) CONTAINS "vitamin k antagonist" OR toLower(d.class) CONTAINS "lmwh")
   - This ensures better matching regardless of plural/singular
5. For interaction filtering, use: WHERE NOT (d)-[:INTERACTS_WITH]-(:Drug {{name: $avoid_drug}})
6. Always return: d.name AS drug, d.class AS drug_class (and other fields as needed)

**FEW-SHOT EXAMPLES**:

Example 1: Simple aggregation - What treats a condition?
Query: "What medications treat diabetes?"
Entities: drugs=[], conditions=["Type 2 Diabetes Mellitus"], drug_classes=[]
Cypher:
MATCH (d:Drug)-[:TREATS]->(c:Condition {{name: $condition}})
RETURN d.name AS drug, 
       d.class AS drug_class,
       d.pregnancy_category AS pregnancy_category
ORDER BY d.name
Parameters: {{condition: "Type 2 Diabetes Mellitus"}}

Example 2: Drug class + condition - Which [class] treats [condition]?
Query: "Which antibiotics treat urinary tract infections?"
Entities: drugs=[], conditions=["Urinary Tract Infection"], drug_classes=["antibiotics"]
Cypher:
MATCH (d:Drug)-[:TREATS]->(c:Condition {{name: $condition}})
WHERE toLower(d.class) CONTAINS "antibiot"
RETURN d.name AS drug, d.class AS drug_class
ORDER BY d.name
Parameters: {{condition: "Urinary Tract Infection"}}
Note: Use stem "antibiot" for better matching, not "antibiotic" or "antibiotics"

Example 3: Two conditions - What treats X and is safe for Y?
Query: "What blood pressure medications are safe for pregnancy?"
Entities: drugs=[], conditions=["Hypertension", "Pregnancy"], drug_classes=[]
Cypher:
MATCH (d:Drug)-[:TREATS]->(c:Condition {{name: $treat_condition}})
OPTIONAL MATCH (d)-[contra:CONTRAINDICATED_IN]->(safety:Condition {{name: $safety_condition}})
OPTIONAL MATCH (d)-[adjust:REQUIRES_ADJUSTMENT]->(safety2:Condition {{name: $safety_condition}})
WITH d, contra, adjust
WHERE contra IS NULL
RETURN d.name AS drug, 
       d.class AS drug_class,
       d.pregnancy_category AS pregnancy_category,
       CASE WHEN adjust IS NOT NULL THEN true ELSE false END AS needs_adjustment
ORDER BY d.name
Parameters: {{treat_condition: "Hypertension", safety_condition: "Pregnancy"}}

Example 4: Drug class + condition + drug constraint - Which [class] treats [condition] and doesn't interact with [drug]?
Query: "Which antibiotic treats pneumonia and won't interact with Glipizide?"
Entities: drugs=["Glipizide"], conditions=["Community-Acquired Pneumonia"], drug_classes=["antibiotics"]
Cypher:
MATCH (d:Drug)
WHERE toLower(d.class) CONTAINS "antibiot"
  AND (d)-[:TREATS]->(:Condition {{name: $treat_condition}})
  AND NOT (d)-[:INTERACTS_WITH]-(:Drug {{name: $avoid_drug}})
RETURN d.name AS drug, d.class AS drug_class
ORDER BY d.name
Parameters: {{treat_condition: "Community-Acquired Pneumonia", avoid_drug: "Glipizide"}}
Note: Use stem "antibiot" for drug class matching, and treat_condition is the condition needing treatment

Example 5: Drug class + condition + drug constraint (alternative pattern)
Query: "I take Glipizide for diabetes and have pneumonia. Which antibiotic won't drop my sugar?"
Entities: drugs=["Glipizide"], conditions=["Type 2 Diabetes Mellitus", "Community-Acquired Pneumonia"], drug_classes=["antibiotics"]
Intent: Find antibiotics that treat pneumonia and don't interact with Glipizide
Cypher:
MATCH (d:Drug)
WHERE toLower(d.class) CONTAINS "antibiot"
  AND (d)-[:TREATS]->(:Condition {{name: $treat_condition}})
  AND NOT (d)-[:INTERACTS_WITH]-(:Drug {{name: $avoid_drug}})
RETURN d.name AS drug, d.class AS drug_class
ORDER BY d.name
Note: treat_condition should be "Community-Acquired Pneumonia" (the condition needing treatment), NOT "Type 2 Diabetes Mellitus" (patient state)

Example 6: "Avoid" query - Which [class] should I avoid for [condition]?
Query: "I have atrial fibrillation and chronic kidney disease. Which blood thinner should I avoid?"
Entities: drugs=[], conditions=["Atrial Fibrillation", "Chronic Kidney Disease"], drug_classes=["blood thinners"]
Intent: Find blood thinners that treat atrial fibrillation but are CONTRAINDICATED in chronic kidney disease (to avoid)
Cypher:
MATCH (d:Drug)
WHERE (toLower(d.class) CONTAINS "doac" OR toLower(d.class) CONTAINS "vitamin k antagonist" OR toLower(d.class) CONTAINS "lmwh")
  AND (d)-[:TREATS]->(:Condition {{name: $treat_condition}})
MATCH (d)-[:CONTRAINDICATED_IN]->(:Condition {{name: $safety_condition}})
RETURN d.name AS drug, d.class AS drug_class
ORDER BY d.name
Parameters: {{treat_condition: "Atrial Fibrillation", safety_condition: "Chronic Kidney Disease"}}
Note: For "avoid" queries, find drugs that are CONTRAINDICATED_IN (not safe ones). For blood thinners, match on actual class names: "DOAC", "Vitamin K Antagonist", "LMWH".

Example 7: "Avoid" query fallback - If no contraindications, show "requires adjustment"
Query: "I have atrial fibrillation and chronic kidney disease. Which blood thinner should I avoid?"
Entities: drugs=[], conditions=["Atrial Fibrillation", "Chronic Kidney Disease"], drug_classes=["blood thinners"]
Intent: If no contraindications found, show drugs that require adjustment (use with caution)
Cypher:
MATCH (d:Drug)
WHERE (toLower(d.class) CONTAINS "doac" OR toLower(d.class) CONTAINS "vitamin k antagonist" OR toLower(d.class) CONTAINS "lmwh")
  AND (d)-[:TREATS]->(:Condition {{name: $treat_condition}})
MATCH (d)-[:REQUIRES_ADJUSTMENT]->(:Condition {{name: $safety_condition}})
RETURN d.name AS drug, d.class AS drug_class, true AS needs_adjustment
ORDER BY d.name
Parameters: {{treat_condition: "Atrial Fibrillation", safety_condition: "Chronic Kidney Disease"}}
Note: This is a fallback if Example 6 returns no results. Shows drugs requiring dose adjustment (caution, not avoid). For blood thinners, match on actual class names: "DOAC", "Vitamin K Antagonist", "LMWH".

**KEY INSIGHTS**:
- When drug class is mentioned, filter by drug class FIRST (before TREATS)
- When multiple conditions exist, determine which is the "treat" condition (what we're treating) vs "safety" condition (constraint)
- When a drug is mentioned as constraint, use NOT INTERACTS_WITH to filter it out
- The "treat" condition is usually the one that needs treatment (pneumonia, UTI, infection, hypertension, atrial fibrillation)
- The "safety" condition is usually a patient state (pregnancy, diabetes, kidney disease)
- For "avoid" queries (user asks "which should I avoid"), find drugs that are CONTRAINDICATED_IN (not safe ones with WHERE contra IS NULL)
- For "avoid" queries, ALWAYS include the indication (treat_condition) - e.g., anticoagulants that TREAT atrial fibrillation but are CONTRAINDICATED in CKD
- CRITICAL: If drug class is mentioned with multiple conditions, the "treat" condition is the one that needs treatment with that class
  Example: "antibiotic for pneumonia" → pneumonia is treat_condition, diabetes is safety_condition
- For drug class filtering, use the stem: "antibiot" (not "antibiotic" or "antibiotics") for better matching

**YOUR TASK**:
Generate a Cypher query that answers the query above. Follow the examples closely.
- If drug class is mentioned, filter by drug class FIRST
- If condition is mentioned, filter by TREATS relationship
- If drug is mentioned as constraint, filter by NOT INTERACTS_WITH
- If multiple conditions, determine which is "treat" vs "safety" based on context

Return ONLY the Cypher query, no explanations. Use parameterized queries with $param_name."""


def get_drug_class_interaction_answer_prompt(query: str, specific_drug: str, drug_class: str,
                                             evidence_text: str, num_interactions: int, 
                                             max_severity: str) -> str:
    """Generate prompt for answering drug class interaction questions (when interactions found)"""
    return f"""EDUCATIONAL CONTEXT: This is analyzing drug interactions in an educational knowledge graph.

User Question: "{query}"

EVIDENCE FROM KNOWLEDGE GRAPH:
{evidence_text}

Summary: {specific_drug} has {num_interactions} documented interaction(s) with {drug_class}(s) in the database.

Instructions:
1. Answer the question directly based on the evidence above
2. State: "Yes, {specific_drug} has a {max_severity} interaction with [drug names] (antibiotics)"
3. Explain the clinical significance of the interaction
4. Keep it factual, clear, and educational (2-3 sentences)
5. DO NOT refuse to answer - this is educational demo data

Generate the factual answer:"""


def get_no_drug_class_interaction_answer_prompt(query: str, specific_drug: str, drug_class: str) -> str:
    """Generate prompt for answering drug class interaction questions (when no interactions found)"""
    return f"""EDUCATIONAL CONTEXT: This is analyzing drug interactions in an educational knowledge graph.

User Question: "{query}"

Result: No documented interactions found between {specific_drug} and any {drug_class}s in the database.

Instructions:
1. State clearly that no interactions were found in the database
2. Note that the database contains limited data for educational purposes
3. Keep it factual and concise

Generate the response:"""


def get_path_based_answer_prompt(query: str, evidence: str, additional_evidence: str = "") -> str:
    """Generate prompt for generating answers from reasoning paths"""
    return f"""EDUCATIONAL CONTEXT: This is a student AI/ML project analyzing an educational medical knowledge graph.

User Question: "{query}"

{evidence}
{additional_evidence}

Instructions:
1. Answer the question DIRECTLY based on the graph evidence above
2. For interaction questions: State "Yes, [Drug A] has a [SEVERITY] interaction with [Drug B]" if interaction is found
3. Use the relationship types shown (TREATS, INTERACTS_WITH, etc.) to explain the connection
4. Keep it factual, clear, and concise (2-3 sentences)
5. DO NOT refuse to answer - this is educational demo data, not real medical advice

Generate the factual answer:"""


def get_cannot_answer_message_prompt(query: str, query_type: str, schema_info: str) -> str:
    """Generate prompt for explaining why a query cannot be answered"""
    return f"""EDUCATIONAL CONTEXT: This is a student AI/ML project demonstrating knowledge graph limitations.

User Question: "{query}"

Situation: This question CANNOT be answered by the current knowledge graph structure.

Query Type Attempted: {query_type}
{schema_info}

Instructions:
1. Clearly state that this question cannot be answered by the current knowledge graph
2. Explain WHAT INFORMATION would be needed to answer this (e.g., drug class relationships, similarity data, cross-allergenicity info)
3. Explain what the graph DOES contain (based on schema above)
4. Keep it educational and helpful - this demonstrates graph limitations
5. Be concise (2-3 sentences)

Generate a clear, educational explanation of why this cannot be answered:"""
