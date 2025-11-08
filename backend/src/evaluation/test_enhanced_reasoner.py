# test_enhanced_reasoner.py
"""
Test script for enhanced beam search reasoner with LLM-guided aggregation
Tests all query types: simple aggregation, filtered aggregation, interaction checks, and path traversal
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.src.graph.beam_searcher import BeamSearchReasoner
from backend.src.evaluation.test_questions import SIMPLE_QUESTIONS, MEDIUM_QUESTIONS, COMPLEX_QUESTIONS

def print_separator(char="=", length=80):
    print(char * length)

def test_query(reasoner, question_obj, show_details=True):
    """Test a single query and display results"""
    question = question_obj["question"]
    q_id = question_obj["id"]
    complexity = question_obj["complexity"]
    
    print(f"\n[{q_id}] {question}")
    print(f"Expected complexity: {complexity}")
    print_separator("-", 80)
    
    try:
        result = reasoner.answer_question(question)
        
        # Display results
        query_type = result.get('query_type', 'unknown')
        print(f"Query Type: {query_type}")
        print()
        print(result['answer'])
        print(f"\nConfidence: {result['confidence']:.1%}")
        
        # Show reasoning paths if available
        if show_details and result.get('paths'):
            print("\nReasoning Paths:")
            for i, p in enumerate(result["paths"][:2], 1):
                print(f"  Path {i} (score: {p['score']:.1f}):")
                print(f"    {' → '.join(p['nodes'])}")
        
        # Show intent if available
        if show_details and result.get('intent'):
            print(f"\nLLM Intent Analysis:\n{result['intent']}")
        
        return True, result
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    print_separator()
    print("ENHANCED MEDICAL REASONING ENGINE - TEST SUITE")
    print("Testing LLM-guided query routing and aggregation")
    print_separator()
    
    reasoner = BeamSearchReasoner(beam_width=3, max_depth=4)
    
    # Test statistics
    total_tests = 0
    successful_tests = 0
    results_by_type = {
        "simple": [],
        "medium": [],
        "complex": []
    }
    
    # Test Simple Questions
    print("\n\n" + "="*80)
    print("TESTING SIMPLE QUESTIONS (1-2 hops)")
    print("="*80)
    
    for q in SIMPLE_QUESTIONS[:3]:  # Test first 3
        total_tests += 1
        success, result = test_query(reasoner, q, show_details=False)
        if success:
            successful_tests += 1
            results_by_type["simple"].append((q["id"], result))
        print()
    
    # Test Medium Questions
    print("\n\n" + "="*80)
    print("TESTING MEDIUM QUESTIONS (2-3 hops, Multi-hop Traversal)")
    print("="*80)
    
    for q in MEDIUM_QUESTIONS[:3]:  # Test first 3
        total_tests += 1
        success, result = test_query(reasoner, q, show_details=True)
        if success:
            successful_tests += 1
            results_by_type["medium"].append((q["id"], result))
        print()
    
    # Test Complex Questions
    print("\n\n" + "="*80)
    print("TESTING COMPLEX QUESTIONS (3-5 hops, Multi-constraint)")
    print("="*80)
    
    for q in COMPLEX_QUESTIONS[:3]:  # Test first 3
        total_tests += 1
        success, result = test_query(reasoner, q, show_details=True)
        if success:
            successful_tests += 1
            results_by_type["complex"].append((q["id"], result))
        print()
    
    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
    
    print("\n\nQuery Type Distribution:")
    all_results = results_by_type["simple"] + results_by_type["medium"] + results_by_type["complex"]
    query_types = {}
    for qid, result in all_results:
        qtype = result.get('query_type', 'unknown')
        query_types[qtype] = query_types.get(qtype, 0) + 1
    
    for qtype, count in query_types.items():
        print(f"  • {qtype}: {count} queries")
    
    print("\n" + "="*80)
    
    reasoner.close()


if __name__ == "__main__":
    main()

