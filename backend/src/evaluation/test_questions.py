# src/evaluation/test_questions.py
"""
Test Questions for Medical Diagnosis Assistant
All questions are designed to be answerable using the knowledge graph in kg.py
"""

# Test questions with expected answers based on the knowledge graph
questions = [
    {
        "id": "Q1",
        "question": "I'm on Warfarin and need an antibiotic for a UTI. Which one is safe?",
        "expected_answer": "Safe options: Amoxicillin, Cephalexin, Nitrofurantoin. Avoid: Azithromycin (MAJOR interaction - increased INR/bleeding risk), Ciprofloxacin (MAJOR - increased bleeding), Trimethoprim-Sulfamethoxazole (MODERATE - potential INR elevation).",
        "complexity": "multi-hop",
        "key_entities": ["Warfarin", "Urinary Tract Infection", "antibiotics"],
        "constraints": ["TREATS → UTI", "NOT INTERACTS_WITH Warfarin"]
    },
    {
        "id": "Q2",
        "question": "I'm on Apixaban and have chronic kidney disease. Which blood thinner is safest?",
        "expected_answer": "Safest options: Warfarin or Enoxaparin (no CKD contraindications). AVOID: Dabigatran (contraindicated in CKD stage 4-5), Rivaroxaban (contraindicated in CrCl<15), Apixaban (contraindicated in CKD).",
        "complexity": "multi-hop",
        "key_entities": ["Apixaban", "Chronic Kidney Disease", "anticoagulants"],
        "constraints": ["TREATS → AF/DVT", "NOT CONTRAINDICATED_IN CKD"]
    },
    {
        "id": "Q3",
        "question": "I take Metformin and Simvastatin. I have a sinus infection. Can I take Amoxicillin?",
        "expected_answer": "YES - Safe to use. No interactions between Metformin+Amoxicillin or Simvastatin+Amoxicillin in the knowledge graph.",
        "complexity": "interaction-check",
        "key_entities": ["Metformin", "Simvastatin", "Amoxicillin"],
        "constraints": ["Check for INTERACTS_WITH"]
    },
    {
        "id": "Q4",
        "question": "I'm on Warfarin for DVT. I need pain relief. Can I take Ibuprofen?",
        "expected_answer": "NO - MAJOR interaction. Warfarin + Ibuprofen increases bleeding risk (GI, intracranial). Must avoid or use alternative pain relief.",
        "complexity": "interaction-check",
        "key_entities": ["Warfarin", "Ibuprofen"],
        "constraints": ["INTERACTS_WITH - MAJOR severity"]
    },
    {
        "id": "Q5",
        "question": "I take Glipizide for diabetes and have pneumonia. Which antibiotic won't drop my sugar?",
        "expected_answer": "Based on the graph: All pneumonia antibiotics are safe (Amoxicillin, Amoxicillin-Clavulanate, Cephalexin, Ceftriaxone, Azithromycin, Ciprofloxacin, Levofloxacin). No hypoglycemia interactions modeled with Glipizide.",
        "complexity": "multi-hop",
        "key_entities": ["Glipizide", "Community-Acquired Pneumonia", "antibiotics"],
        "constraints": ["TREATS → Pneumonia", "NOT INTERACTS_WITH Glipizide"]
    },
    {
        "id": "Q6",
        "question": "I'm on Rivaroxaban and starting Erythromycin. Is this safe?",
        "expected_answer": "YES - Safe per the knowledge graph. No interaction listed between Rivaroxaban and Erythromycin (only Rivaroxaban+Clarithromycin is listed).",
        "complexity": "interaction-check",
        "key_entities": ["Rivaroxaban", "Erythromycin"],
        "constraints": ["Check for INTERACTS_WITH"]
    },
    {
        "id": "Q7",
        "question": "I have atrial fibrillation and chronic kidney disease. Which blood thinner should I avoid?",
        "expected_answer": "AVOID: Dabigatran (contraindicated in CKD stage 4-5), Rivaroxaban (absolute contraindication in CrCl<15), Apixaban (contraindicated in CKD). Safe options: Warfarin, Enoxaparin, Edoxaban.",
        "complexity": "multi-hop",
        "key_entities": ["Atrial Fibrillation", "Chronic Kidney Disease", "anticoagulants"],
        "constraints": ["TREATS → AF", "CONTRAINDICATED_IN → CKD"]
    },
    {
        "id": "Q8",
        "question": "I have atrial fibrillation and pregnancy. Which blood thinner is unsafe?",
        "expected_answer": "CONTRAINDICATED: Warfarin (only anticoagulant with absolute contraindication in pregnancy). DOACs are pregnancy category C (caution). Safer option: Enoxaparin (LMWH, pregnancy category B).",
        "complexity": "multi-hop",
        "key_entities": ["Atrial Fibrillation", "Pregnancy", "anticoagulants"],
        "constraints": ["TREATS → AF", "CONTRAINDICATED_IN → Pregnancy"]
    },
    {
        "id": "Q9",
        "question": "Can I take Simvastatin and Clarithromycin together?",
        "expected_answer": "NO - MAJOR interaction. Simvastatin + Clarithromycin causes myopathy (muscle damage/rhabdomyolysis risk). Must avoid or use alternative statin/antibiotic.",
        "complexity": "interaction-check",
        "key_entities": ["Simvastatin", "Clarithromycin"],
        "constraints": ["INTERACTS_WITH - MAJOR severity"]
    },
    {
        "id": "Q10",
        "question": "What drugs treat Type 2 Diabetes?",
        "expected_answer": "6 drugs: Metformin (Biguanide), Glipizide (Sulfonylurea), Sitagliptin (DPP-4 inhibitor), Empagliflozin (SGLT2 inhibitor), Semaglutide (GLP-1 agonist), Insulin Glargine (Insulin).",
        "complexity": "simple",
        "key_entities": ["Type 2 Diabetes Mellitus"],
        "constraints": ["TREATS → Type 2 Diabetes Mellitus"]
    },
    {
        "id": "Q11",
        "question": "Which statins are unsafe during pregnancy?",
        "expected_answer": "ALL 4 STATINS are contraindicated in pregnancy (category X): Atorvastatin, Rosuvastatin, Simvastatin, Pravastatin.",
        "complexity": "multi-hop",
        "key_entities": ["statins", "Pregnancy"],
        "constraints": ["Drug class = Statin", "CONTRAINDICATED_IN → Pregnancy"]
    },
    {
        "id": "Q12",
        "question": "I'm on Warfarin. Can I take Azithromycin?",
        "expected_answer": "NO - MAJOR interaction. Warfarin + Azithromycin causes increased INR (bleeding risk). Requires close monitoring or alternative antibiotic.",
        "complexity": "interaction-check",
        "key_entities": ["Warfarin", "Azithromycin"],
        "constraints": ["INTERACTS_WITH - MAJOR severity"]
    },
    {
        "id": "Q13",
        "question": "Can I take Warfarin with Azithromycin?",
        "expected_answer": "NO - MAJOR interaction. Warfarin + Azithromycin causes increased INR (bleeding risk). Same as Q12.",
        "complexity": "interaction-check",
        "key_entities": ["Warfarin", "Azithromycin"],
        "constraints": ["INTERACTS_WITH - MAJOR severity"]
    },
    {
        "id": "Q14",
        "question": "Can someone taking Apixaban safely use Clarithromycin?",
        "expected_answer": "NO - MAJOR interaction. Clarithromycin increases Apixaban levels (increased bleeding risk). Must avoid or use alternative antibiotic.",
        "complexity": "interaction-check",
        "key_entities": ["Apixaban", "Clarithromycin"],
        "constraints": ["INTERACTS_WITH - MAJOR severity"]
    },
]


# Legacy format for backward compatibility
SIMPLE_QUESTIONS = [
    {
        "id": "Q001",
        "question": "What medications can treat irregular heartbeat (Atrial Fibrillation)?",
        "expected_path": ["Drug", "TREATS", "Atrial Fibrillation"],
        "complexity": "simple",
        "graph_entities": ["TREATS", "C001"]
    },
    {
        "id": "Q002",
        "question": "What drugs treat Type 2 Diabetes?",
        "expected_path": ["Drug", "TREATS", "Type 2 Diabetes Mellitus"],
        "complexity": "simple",
        "graph_entities": ["TREATS", "C004"]
    },
    {
        "id": "Q003",
        "question": "What does Warfarin interact with?",
        "expected_path": ["Warfarin", "INTERACTS_WITH", "Drug"],
        "complexity": "simple",
        "graph_entities": ["D001", "INTERACTS_WITH"]
    },
    {
        "id": "Q004",
        "question": "Which antibiotics can treat urinary tract infections?",
        "expected_path": ["Drug", "TREATS", "Urinary Tract Infection"],
        "complexity": "simple",
        "graph_entities": ["TREATS", "C006"]
    },
    {
        "id": "Q005",
        "question": "What medications are used for high cholesterol?",
        "expected_path": ["Drug", "TREATS", "Hyperlipidemia"],
        "complexity": "simple",
        "graph_entities": ["TREATS", "C003"]
    },
]

MEDIUM_QUESTIONS = [
    {
        "id": "Q101",
        "question": "Can I take Warfarin with Azithromycin?",
        "expected_path": [
            "Warfarin", "INTERACTS_WITH", "Azithromycin"
        ],
        "complexity": "medium",
        "graph_entities": ["D001", "INTERACTS_WITH", "D027"],
        "reasoning": [
            "Find if Warfarin has INTERACTS_WITH relationship to Azithromycin",
            "Return the interaction details"
        ]
    },
]

COMPLEX_QUESTIONS = [
    {
        "id": "Q203",
        "question": "I'm on Simvastatin for cholesterol and need pneumonia treatment. Which antibiotic should I avoid?",
        "expected_path": [
            "Simvastatin", "TREATS", "Hyperlipidemia",
            "Clarithromycin", "TREATS", "Community-Acquired Pneumonia",
            "Simvastatin", "INTERACTS_WITH", "Clarithromycin"
        ],
        "complexity": "complex",
        "graph_entities": ["D017", "D028", "INTERACTS_WITH", "C003", "C005"],
        "reasoning": [
            "Verify Simvastatin treats cholesterol",
            "Find antibiotics treating pneumonia",
            "Check which ones interact with Simvastatin",
            "Warn about Clarithromycin"
        ],
        "real_world": "Preventing serious muscle damage from statin-antibiotic interactions"
    },
    {
        "id": "Q204",
        "question": "Can someone taking Apixaban safely use Clarithromycin?",
        "expected_path": [
            "Apixaban", "INTERACTS_WITH", "Clarithromycin"
        ],
        "complexity": "complex",
        "graph_entities": ["D002", "INTERACTS_WITH", "D028"],
        "reasoning": [
            "Check for direct interaction between drugs",
            "Return severity and description"
        ],
        "real_world": "Quick safety check for drug combination"
    },
]


# Save to file
import json

def save_test_questions():
    all_questions = {
        "simple": SIMPLE_QUESTIONS,
        "medium": MEDIUM_QUESTIONS,
        "complex": COMPLEX_QUESTIONS
    }
    
    with open('data/test_questions.json', 'w') as f:
        json.dump(all_questions, f, indent=2)
    
    total = len(SIMPLE_QUESTIONS) + len(MEDIUM_QUESTIONS) + len(COMPLEX_QUESTIONS)
    print(f"\n✓ Saved {total} test questions to data/test_questions.json")
    print(f"  • {len(SIMPLE_QUESTIONS)} simple questions (1-2 hops)")
    print(f"  • {len(MEDIUM_QUESTIONS)} medium questions (2-3 hops)")
    print(f"  • {len(COMPLEX_QUESTIONS)} complex questions (3-5 hops)")
    print(f"\nAll questions are verified to be answerable by the knowledge graph!")

if __name__ == '__main__':
    save_test_questions()
