# src/reasoning/llm_agent.py
"""
LLM-guided action selector for Think-on-Graph beam search.
Uses the globally-initialized AzureChatOpenAI instance (`llm`) from the project.
"""

import logging
from typing import List, Dict

# Import the **single** LLM instance you created in the root of the repo
try:
    from backend.src.utils.llm_config import llm
except ModuleNotFoundError:
    from src.utils.llm_config import llm

logger = logging.getLogger(__name__)

class LLMAgent:
    """
    Thin wrapper that asks the LLM which neighbor(s) to explore next.
    Returns up to `max_choices` neighbor dicts.
    """

    def __init__(self, max_choices: int = 3):
        self.max_choices = max_choices

    def propose_action(
        self,
        path_nodes: List[str],
        query: str,
        neighbors: List[Dict],
    ) -> List[Dict]:
        """
        Args:
            path_nodes:  list of node **names** in the current path (e.g. ["Metformin"])
            query:       original user question
            neighbors:   list of neighbor dicts from `get_neighbors()`

        Returns:
            List of selected neighbor dicts (max `self.max_choices`)
        """
        if not neighbors:
            return []

        # ------------------------------------------------------------------
        # 1. Build a compact, readable prompt
        # ------------------------------------------------------------------
        path_str = " → ".join(path_nodes) if len(path_nodes) > 1 else path_nodes[0]

        # Show at most 12 options to keep token count low
        options = []
        for i, n in enumerate(neighbors[:12], start=1):
            rel = n["relationship"]
            target = n["name"]
            options.append(f"{i}. {path_nodes[-1]} {rel} {target}")

        prompt = f"""You are a medical reasoning engine.  
Query: "{query}"
Current reasoning path: {path_str}

Possible next steps (choose the most relevant to answer the query):
{chr(10).join(options)}

Priorities (in order):
1. TREATS
2. CONTRAINDICATED_IN / REQUIRES_ADJUSTMENT / CONTRAINDICATES
3. INTERACTS_WITH
4. Any other medically meaningful relation

Return **only the numbers** of the best next steps, comma-separated (e.g. "1,3").
If none are useful, reply "NONE".
"""

        # ------------------------------------------------------------------
        # 2. Call Azure LLM (LangChain wrapper)
        # ------------------------------------------------------------------
        try:
            from langchain_core.messages import HumanMessage
            
            # Use LangChain message format
            response = llm.invoke([HumanMessage(content=prompt)])
            choice_text = response.content.strip()
            logger.info(f"LLM raw response: {choice_text!r}")

            # ------------------------------------------------------------------
            # 3. Parse numbers → neighbor dicts
            # ------------------------------------------------------------------
            if choice_text.upper() == "NONE":
                return []                                 # LLM says stop / no good hop

            selected_idxs = []
            for token in choice_text.replace(",", " ").split():
                if token.isdigit():
                    idx = int(token) - 1                # 1-based → 0-based
                    if 0 <= idx < len(neighbors):
                        selected_idxs.append(idx)

            # Deduplicate & respect max_choices
            selected_idxs = list(dict.fromkeys(selected_idxs))[: self.max_choices]

            result = [neighbors[i] for i in selected_idxs]
            logger.info(f"LLM selected {len(result)} neighbor(s)")
            return result

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Fallback: pick first few neighbors that look medical
            fallback = [
                n for n in neighbors
                if n["relationship"] in {
                    "TREATS",
                    "CONTRAINDICATED_IN", "REQUIRES_ADJUSTMENT", "CONTRAINDICATES",
                    "INTERACTS_WITH"
                }
            ][: self.max_choices]
            if not fallback:
                fallback = neighbors[: self.max_choices]
            return fallback