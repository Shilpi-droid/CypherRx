# src/graph/relationship_direction.py
"""
Shared helpers for phrasing graph relationships correctly regardless of which
way a traversal walked the edge. Relationships in the KG are only ever created
in one direction (e.g. Drug -[:TREATS]-> Condition); when beam search or the
LLM agent walks the edge backward (Condition -> Drug), naively saying
"Condition TREATS Drug" would be false. `direction` ('outgoing'/'incoming')
comes from `BeamSearchReasoner.get_neighbors`.
"""

REVERSE_RELATIONSHIP_LABELS = {
    "TREATS": "TREATED_BY",
    "REQUIRES_ADJUSTMENT": "ADJUSTMENT_REQUIRED_FOR",
}


def directional_relationship(relationship: str, direction: str) -> str:
    """Return the relationship label as it should read for the given traversal direction."""
    if direction == "incoming":
        return REVERSE_RELATIONSHIP_LABELS.get(relationship, relationship)
    return relationship


def directional_phrase(subject: str, relationship: str, obj: str, direction: str) -> str:
    """Build a grammatically correct 'subject relationship object' phrase for a traversed edge.

    `subject` is the node being expanded from and `obj` is the neighbor; if the edge
    actually points obj -> subject in the graph, the sentence order is swapped so it
    still reads as a true statement.
    """
    if direction == "incoming":
        return f"{obj} {relationship} {subject}"
    return f"{subject} {relationship} {obj}"
