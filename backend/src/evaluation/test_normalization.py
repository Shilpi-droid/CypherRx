#!/usr/bin/env python3
"""
Test script to verify entity normalization in QueryPlanner
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.src.graph.query_planner import QueryPlanner


def test_normalization():
    """Test entity normalization with all test questions"""
    
    planner = QueryPlanner()
    
    test_cases = [
        # From test_questions.py
        {
            "original": "What are the side effects of Aspirin?",
            "expected_entities": ["Aspirin"],
        },
        {
            "original": "What drugs treat Type 2 Diabetes?",
            "expected_entities": ["Type 2 Diabetes"],
        },
        {
            "original": "What does Warfarin interact with?",
            "expected_entities": ["Warfarin"],
        },
        {
            "original": "Which drugs for hypertension don't cause dizziness?",
            "expected_entities": ["Hypertension", "Dizziness"],
        },
        {
            "original": "What are safe blood thinners that don't interact with Aspirin?",
            "expected_entities": ["Anticoagulant", "Aspirin"],
            "colloquial": ["blood thinners"]
        },
        {
            "original": "Which diabetes medications target insulin pathways?",
            "expected_entities": ["Diabetes", "Insulin"],
        },
        {
            "original": "What diabetes drugs are safe for patients with kidney disease taking blood thinners?",
            "expected_entities": ["Diabetes", "Kidney disease", "Anticoagulant"],
            "colloquial": ["blood thinners"]
        },
        {
            "original": "Which hypertension medications don't cause fatigue and are safe with antidepressants?",
            "expected_entities": ["Hypertension", "Fatigue", "SSRI"],
            "colloquial": ["antidepressants"]
        },
        {
            "original": "What are alternative antibiotics for strep throat that don't interact with birth control?",
            "expected_entities": ["Antibiotic", "Streptococcal infection", "Contraceptive"],
            "colloquial": ["antibiotics", "strep throat", "birth control"]
        },
    ]
    
    print("=" * 80)
    print("ENTITY NORMALIZATION TEST")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        original = test_case["original"]
        normalized = planner._normalize_query(original)
        
        print(f"Test Case {i}:")
        print(f"  Original:   {original}")
        print(f"  Normalized: {normalized}")
        
        # Check if expected entities are in normalized query
        all_found = True
        for entity in test_case["expected_entities"]:
            if entity in normalized:
                print(f"    ✅ Found: {entity}")
            else:
                print(f"    ❌ Missing: {entity}")
                all_found = False
        
        # Check if colloquial terms were replaced
        if "colloquial" in test_case:
            for colloquial in test_case["colloquial"]:
                if colloquial.lower() not in normalized.lower():
                    print(f"    ✅ Replaced: {colloquial}")
                else:
                    print(f"    ⚠️  Still present: {colloquial}")
        
        if all_found:
            print("  Status: PASSED ✅")
            passed += 1
        else:
            print("  Status: FAILED ❌")
            failed += 1
        
        print()
    
    print("=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    return failed == 0


def test_specific_mappings():
    """Test specific entity mappings"""
    
    planner = QueryPlanner()
    
    print("\n" + "=" * 80)
    print("SPECIFIC MAPPING TESTS")
    print("=" * 80)
    print()
    
    mappings_to_test = [
        ("strep throat", "Streptococcal infection"),
        ("birth control", "Contraceptive"),
        ("blood thinners", "Anticoagulant"),
        ("high blood pressure", "Hypertension"),
        ("type 2 diabetes", "Type 2 Diabetes"),
        ("antidepressants", "SSRI"),
        ("dizzy", "Dizziness"),
        ("tired", "Fatigue"),
    ]
    
    passed = 0
    failed = 0
    
    for colloquial, expected_formal in mappings_to_test:
        test_query = f"What about {colloquial}?"
        normalized = planner._normalize_query(test_query)
        
        if expected_formal in normalized:
            print(f"✅ '{colloquial}' → '{expected_formal}'")
            passed += 1
        else:
            print(f"❌ '{colloquial}' → Expected '{expected_formal}' but got: {normalized}")
            failed += 1
    
    print()
    print(f"Mapping Tests: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    # Run tests
    test1_passed = test_normalization()
    test2_passed = test_specific_mappings()
    
    # Overall result
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED")
        sys.exit(1)

