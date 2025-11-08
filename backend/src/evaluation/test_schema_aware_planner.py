"""
Test script to verify schema-aware query planning improvements.
"""

import os
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from backend.src.graph.query_planner import QueryPlanner
from backend.src.graph.schema_inspector import GraphSchemaInspector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


def test_schema_inspection():
    """Test that schema inspection works correctly."""
    print("\n" + "="*70)
    print("TEST 1: Schema Inspection")
    print("="*70)

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("DATABASE_NAME", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        inspector = GraphSchemaInspector(driver, database)
        schema = inspector.get_schema()

        print("\n[SCHEMA] Database Schema Summary:")
        print(schema.to_summary())

        print("\n[SUCCESS] Schema inspection successful!")
        return driver, schema

    except Exception as e:
        print(f"\n[ERROR] Schema inspection failed: {e}")
        driver.close()
        raise


def test_query_planning_with_schema(driver):
    """Test query planning with schema awareness."""
    print("\n" + "="*70)
    print("TEST 2: Schema-Aware Query Planning")
    print("="*70)

    database = os.getenv("DATABASE_NAME", "neo4j")
    planner = QueryPlanner(driver=driver, database=database)

    # Test queries
    test_queries = [
        "What are alternative antibiotics for strep throat that don't interact with birth control?",
        "Which drugs treat Hypertension?",
        "What does Warfarin interact with?",
        "Side effects of Aspirin?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[QUERY] Test Query {i}: {query}")
        print("-" * 70)

        try:
            cypher = planner.plan(query)
            print(f"[OK] Generated Cypher:\n{cypher}\n")

            # Validate the query
            if planner.schema_inspector:
                validation = planner.schema_inspector.validate_cypher_entities(cypher)
                if validation["valid"]:
                    print("[OK] Schema validation passed!")
                else:
                    print("[WARN] Schema validation issues:")
                    for issue in validation["issues"]:
                        print(f"  - {issue}")
                    for suggestion in validation["suggestions"]:
                        print(f"  [HINT] {suggestion}")

            # Try executing the query
            with driver.session(database=database) as session:
                result = session.run(cypher)
                records = list(result)
                print(f"[SCHEMA] Query returned {len(records)} results")
                if records:
                    print(f"   Sample results: {records[:3]}")

        except Exception as e:
            print(f"[ERROR] Query planning failed: {e}")

    print("\n[OK] Query planning tests complete!")


def test_comparison_without_schema():
    """Compare query planning without schema awareness."""
    print("\n" + "="*70)
    print("TEST 3: Query Planning WITHOUT Schema (for comparison)")
    print("="*70)

    planner_no_schema = QueryPlanner()  # No driver passed

    query = "What are alternative antibiotics for strep throat that don't interact with birth control?"
    print(f"\n[QUERY] Test Query: {query}")
    print("-" * 70)

    try:
        cypher = planner_no_schema.plan(query)
        print(f"Generated Cypher (no schema info):\n{cypher}\n")
        print("[WARN] Note: This query may use non-existent relationships/nodes")
    except Exception as e:
        print(f"[ERROR] Query planning failed: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("[TEST] TESTING SCHEMA-AWARE QUERY PLANNER")
    print("="*70)

    try:
        # Test 1: Schema inspection
        driver, schema = test_schema_inspection()

        # Test 2: Schema-aware query planning
        test_query_planning_with_schema(driver)

        # Test 3: Comparison without schema
        test_comparison_without_schema()

        print("\n" + "="*70)
        print("[OK] ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")

        driver.close()

    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        print("\n" + "="*70)
        print("[ERROR] TEST SUITE FAILED")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
