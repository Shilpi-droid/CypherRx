# src/data_loaders/hardcode_kg.py
"""
Hard-coded knowledge graph for the Think-on-Graph medical demo.
Populates Neo4j with 60 drugs, conditions, contraindications,
drug-drug interactions, and dosing notes.

Run once (or after a DB wipe):
    python -m src.data_loaders.hardcode_kg
"""

from neo4j import GraphDatabase
import json
import pathlib
import os
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ----------------------------------------------------------------------
# Neo4j connection – using .env configuration
# ----------------------------------------------------------------------
URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
USER = os.getenv('NEO4J_USER', 'neo4j')
PASSWORD = os.getenv('NEO4J_PASSWORD', 'neo4j_password')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'neo4j')

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


# ----------------------------------------------------------------------
# Helper: run a Cypher statement in a transaction
# ----------------------------------------------------------------------
def tx_run(tx, cypher: str, **params):
    tx.run(cypher, **params)


# ----------------------------------------------------------------------
# Data definitions
# ----------------------------------------------------------------------
# 1. Drugs (60) – each entry has a stable `drug_id`
DRUGS = [
    # ---- Anticoagulants (6) -------------------------------------------------
    {"drug_id": "D001", "name": "Warfarin",          "class": "Vitamin K Antagonist", "pregnancy": "D"},
    {"drug_id": "D002", "name": "Apixaban",         "class": "DOAC",                "pregnancy": "C"},
    {"drug_id": "D003", "name": "Rivaroxaban",      "class": "DOAC",                "pregnancy": "C"},
    {"drug_id": "D004", "name": "Dabigatran",       "class": "DOAC",                "pregnancy": "C"},
    {"drug_id": "D005", "name": "Edoxaban",         "class": "DOAC",                "pregnancy": "C"},
    {"drug_id": "D006", "name": "Enoxaparin",       "class": "LMWH",                "pregnancy": "B"},

    # ---- Antihypertensives (8) ---------------------------------------------
    {"drug_id": "D007", "name": "Lisinopril",       "class": "ACEI",                "pregnancy": "D"},
    {"drug_id": "D008", "name": "Losartan",        "class": "ARB",                 "pregnancy": "D"},
    {"drug_id": "D009", "name": "Amlodipine",      "class": "CCB",                 "pregnancy": "C"},
    {"drug_id": "D010", "name": "Metoprolol",      "class": "Beta-blocker",        "pregnancy": "C"},
    {"drug_id": "D011", "name": "Hydrochlorothiazide", "class": "Thiazide",        "pregnancy": "B"},
    {"drug_id": "D012", "name": "Spironolactone",   "class": "Potassium-sparing",   "pregnancy": "C"},
    {"drug_id": "D013", "name": "Furosemide",      "class": "Loop diuretic",       "pregnancy": "C"},
    {"drug_id": "D014", "name": "Carvedilol",      "class": "Beta-blocker",        "pregnancy": "C"},

    # ---- Statins (4) --------------------------------------------------------
    {"drug_id": "D015", "name": "Atorvastatin",    "class": "Statin",              "pregnancy": "X"},
    {"drug_id": "D016", "name": "Rosuvastatin",    "class": "Statin",              "pregnancy": "X"},
    {"drug_id": "D017", "name": "Simvastatin",     "class": "Statin",              "pregnancy": "X"},
    {"drug_id": "D018", "name": "Pravastatin",     "class": "Statin",              "pregnancy": "X"},

    # ---- Antibiotics – Penicillins (4) --------------------------------------
    {"drug_id": "D019", "name": "Amoxicillin",      "class": "Penicillin",          "pregnancy": "B"},
    {"drug_id": "D020", "name": "Amoxicillin-Clavulanate", "class": "Penicillin", "pregnancy": "B"},
    {"drug_id": "D021", "name": "Penicillin V",    "class": "Penicillin",          "pregnancy": "B"},
    {"drug_id": "D022", "name": "Piperacillin-Tazobactam", "class": "Penicillin", "pregnancy": "B"},

    # ---- Antibiotics – Cephalosporins (4) -----------------------------------
    {"drug_id": "D023", "name": "Cephalexin",      "class": "Cephalosporin",       "pregnancy": "B"},
    {"drug_id": "D024", "name": "Ceftriaxone",     "class": "Cephalosporin",       "pregnancy": "B"},
    {"drug_id": "D025", "name": "Cefuroxime",      "class": "Cephalosporin",       "pregnancy": "B"},
    {"drug_id": "D026", "name": "Cefepime",        "class": "Cephalosporin",       "pregnancy": "B"},

    # ---- Antibiotics – Macrolides (3) ---------------------------------------
    {"drug_id": "D027", "name": "Azithromycin",    "class": "Macrolide",           "pregnancy": "B"},
    {"drug_id": "D028", "name": "Clarithromycin",  "class": "Macrolide",           "pregnancy": "C"},
    {"drug_id": "D029", "name": "Erythromycin",    "class": "Macrolide",           "pregnancy": "B"},

    # ---- Antibiotics – Fluoroquinolones (3) ---------------------------------
    {"drug_id": "D030", "name": "Ciprofloxacin",   "class": "Fluoroquinolone",     "pregnancy": "C"},
    {"drug_id": "D031", "name": "Levofloxacin",    "class": "Fluoroquinolone",     "pregnancy": "C"},
    {"drug_id": "D032", "name": "Moxifloxacin",    "class": "Fluoroquinolone",     "pregnancy": "C"},

    # ---- Other Antibiotics (4) ---------------------------------------------
    {"drug_id": "D033", "name": "Doxycycline",     "class": "Tetracycline",        "pregnancy": "D"},
    {"drug_id": "D034", "name": "Metronidazole",   "class": "Nitroimidazole",      "pregnancy": "B"},
    {"drug_id": "D035", "name": "Trimethoprim-Sulfamethoxazole", "class": "Sulfonamide", "pregnancy": "C"},
    {"drug_id": "D036", "name": "Vancomycin",     "class": "Glycopeptide",        "pregnancy": "B"},

    # ---- Diabetes (6) -------------------------------------------------------
    {"drug_id": "D037", "name": "Metformin",       "class": "Biguanide",           "pregnancy": "B"},
    {"drug_id": "D038", "name": "Glipizide",       "class": "Sulfonylurea",        "pregnancy": "C"},
    {"drug_id": "D039", "name": "Sitagliptin",     "class": "DPP-4 inhibitor",     "pregnancy": "B"},
    {"drug_id": "D040", "name": "Empagliflozin",   "class": "SGLT2 inhibitor",     "pregnancy": "C"},
    {"drug_id": "D041", "name": "Semaglutide",     "class": "GLP-1 agonist",       "pregnancy": "C"},
    {"drug_id": "D042", "name": "Insulin Glargine","class": "Insulin",             "pregnancy": "B"},

    # ---- CNS – Antidepressants (4) -----------------------------------------
    {"drug_id": "D043", "name": "Sertraline",      "class": "SSRI",                "pregnancy": "C"},
    {"drug_id": "D044", "name": "Fluoxetine",      "class": "SSRI",                "pregnancy": "C"},
    {"drug_id": "D045", "name": "Venlafaxine",     "class": "SNRI",                "pregnancy": "C"},
    {"drug_id": "D046", "name": "Escitalopram",    "class": "SSRI",                "pregnancy": "C"},

    # ---- CNS – Antipsychotics / Others (4) ---------------------------------
    {"drug_id": "D047", "name": "Risperidone",     "class": "Atypical Antipsychotic","pregnancy": "C"},
    {"drug_id": "D048", "name": "Quetiapine",      "class": "Atypical Antipsychotic","pregnancy": "C"},
    {"drug_id": "D049", "name": "Levetiracetam",   "class": "Antiepileptic",       "pregnancy": "C"},
    {"drug_id": "D050", "name": "Gabapentin",      "class": "Anticonvulsant",      "pregnancy": "C"},

    # ---- Miscellaneous (10) ------------------------------------------------
    {"drug_id": "D051", "name": "Aspirin",         "class": "Antiplatelet",        "pregnancy": "C"},
    {"drug_id": "D052", "name": "Clopidogrel",     "class": "Antiplatelet",        "pregnancy": "B"},
    {"drug_id": "D053", "name": "Omeprazole",      "class": "PPI",                 "pregnancy": "C"},
    {"drug_id": "D054", "name": "Albuterol",       "class": "SABA",                "pregnancy": "C"},
    {"drug_id": "D055", "name": "Montelukast",     "class": "LTRA",                "pregnancy": "B"},
    {"drug_id": "D056", "name": "Prednisone",      "class": "Corticosteroid",      "pregnancy": "C"},
    {"drug_id": "D057", "name": "Levothyroxine",   "class": "Thyroid Hormone",     "pregnancy": "A"},
    {"drug_id": "D058", "name": "Allopurinol",     "class": "Xanthine Oxidase Inhibitor","pregnancy": "C"},
    {"drug_id": "D059", "name": "Tamsulosin",      "class": "Alpha-blocker",       "pregnancy": "B"},
    {"drug_id": "D060", "name": "Sildenafil",      "class": "PDE5 inhibitor",      "pregnancy": "B"},
    {"drug_id": "D061", "name": "Ibuprofen", "class": "NSAID", "pregnancy": "C"},
    
    # ---- Additional Antibiotics for UTI (2) ---------------------------------
    {"drug_id": "D062", "name": "Nitrofurantoin",  "class": "Nitrofuran",          "pregnancy": "B"},
    {"drug_id": "D063", "name": "Trimethoprim",    "class": "Antifolate",          "pregnancy": "C"},
    
]

# 2. Conditions (high-frequency)
CONDITIONS = [
    {"cond_id": "C001", "name": "Atrial Fibrillation"},
    {"cond_id": "C002", "name": "Hypertension"},
    {"cond_id": "C003", "name": "Hyperlipidemia"},
    {"cond_id": "C004", "name": "Type 2 Diabetes Mellitus"},
    {"cond_id": "C005", "name": "Community-Acquired Pneumonia"},
    {"cond_id": "C006", "name": "Urinary Tract Infection"},
    {"cond_id": "C007", "name": "Chronic Kidney Disease"},
    {"cond_id": "C008", "name": "Pregnancy"},
    {"cond_id": "C009", "name": "Deep Vein Thrombosis"},
    {"cond_id": "C010", "name": "Major Depressive Disorder"},
    {"cond_id": "C011", "name": "Asthma"},
    {"cond_id": "C012", "name": "COPD"},
    {"cond_id": "C013", "name": "Gout"},
    {"cond_id": "C014", "name": "Benign Prostatic Hyperplasia"},
    {"cond_id": "C015", "name": "Erectile Dysfunction"},
]

# 3. TREATS relationships (drug → condition)
TREATS = [
    # Anticoagulants
    ("D001", "C001"), ("D002", "C001"), ("D003", "C001"), ("D004", "C001"), ("D005", "C001"),
    ("D001", "C009"), ("D002", "C009"), ("D003", "C009"), ("D004", "C009"), ("D005", "C009"),
    # Antihypertensives
    ("D007", "C002"), ("D008", "C002"), ("D009", "C002"), ("D010", "C002"), ("D011", "C002"),
    ("D012", "C002"), ("D013", "C002"), ("D014", "C002"),
    # Statins
    ("D015", "C003"), ("D016", "C003"), ("D017", "C003"), ("D018", "C003"),
    # Diabetes
    ("D037", "C004"), ("D038", "C004"), ("D039", "C004"), ("D040", "C004"), ("D041", "C004"), ("D042", "C004"),
    # Pneumonia antibiotics
    ("D019", "C005"), ("D020", "C005"), ("D023", "C005"), ("D024", "C005"), ("D027", "C005"),
    ("D030", "C005"), ("D031", "C005"),
    # UTI antibiotics
    ("D019", "C006"), ("D023", "C006"), ("D027", "C006"), ("D030", "C006"), ("D035", "C006"),
    # Asthma / COPD
    ("D054", "C011"), ("D054", "C012"), ("D055", "C011"),
    # Depression
    ("D043", "C010"), ("D044", "C010"), ("D045", "C010"), ("D046", "C010"),
    # Gout
    ("D058", "C013"),
    # BPH
    ("D059", "C014"),
    # ED
    ("D060", "C015"),
    # Additional UTI antibiotics (new)
    ("D062", "C006"),  # Nitrofurantoin → TREATS → Urinary Tract Infection
    ("D063", "C006"),  # Trimethoprim → TREATS → Urinary Tract Infection
]

# 4. CONTRAINDICATED_IN (absolute)
CONTRAINDICATED = [
    # DOACs in severe renal impairment
    ("D004", "C007"),   # Dabigatran contraindicated in CKD stage 4-5
    ("D003", "C007"),   # Rivaroxaban dose-adjust but absolute in CrCl<15
    ("D002", "C007"),   # Apixaban OK up to CrCl 15
    # Fluoroquinolones in pregnancy
    ("D030", "C008"), ("D031", "C008"), ("D032", "C008"),
    # Warfarin in pregnancy
    ("D001", "C008"),
    # Statins in pregnancy
    ("D015", "C008"), ("D016", "C008"), ("D017", "C008"), ("D018", "C008"),
    # Trimethoprim-Sulfa in 1st trimester
    ("D035", "C008"),
]

# 5. REQUIRES_ADJUSTMENT (dose change in CKD)
REQUIRES_ADJUST = [
    ("D002", "C007"),   # Apixaban: 2.5 mg BID if CrCl<30 or age≥80 or weight≤60
    ("D003", "C007"),   # Rivaroxaban: 15 mg daily if CrCl 15-49
    ("D004", "C007"),   # Dabigatran: 75 mg BID if CrCl 15-30
    ("D037", "C007"),   # Metformin: avoid if eGFR<30
]

# 6. Drug-Drug interactions (severity: MAJOR / MODERATE)
INTERACTIONS = [
    # Warfarin interactions
    ("D001", "D027", "MAJOR",   "Increased INR → bleeding risk"),
    ("D001", "D030", "MAJOR",   "Increased bleeding risk"),
    ("D001", "D035", "MODERATE","Potential INR elevation"),
    # DOAC + strong P-gp/CYP3A4 inhibitors
    ("D002", "D028", "MAJOR",   "Clarithromycin → ↑ apixaban"),
    ("D003", "D028", "MAJOR",   "Clarithromycin → ↑ rivaroxaban"),
    # Statin + macrolide
    ("D017", "D028", "MAJOR",   "Simvastatin + clarithromycin → myopathy"),
    ("D001", "D061", "MAJOR", "Increased bleeding risk (GI, intracranial)"),
]

# ----------------------------------------------------------------------
# Build the graph
# ----------------------------------------------------------------------
def wipe_db(tx):
    tx.run("MATCH (n) DETACH DELETE n")

def create_constraints(tx):
    tx.run("CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (d:Drug) REQUIRE d.drug_id IS UNIQUE")
    tx.run("CREATE CONSTRAINT cond_id IF NOT EXISTS FOR (c:Condition) REQUIRE c.cond_id IS UNIQUE")

def load_drugs(tx):
    cypher = """
    UNWIND $drugs AS drug
    CREATE (d:Drug {
        drug_id: drug.drug_id,
        name: drug.name,
        class: drug.class,
        pregnancy_category: drug.pregnancy
    })
    """
    tx.run(cypher, drugs=DRUGS)

def load_conditions(tx):
    cypher = """
    UNWIND $conds AS cond
    CREATE (c:Condition {cond_id: cond.cond_id, name: cond.name})
    """
    tx.run(cypher, conds=CONDITIONS)

def load_treats(tx):
    cypher = """
    UNWIND $rels AS r
    MATCH (d:Drug {drug_id: r[0]}), (c:Condition {cond_id: r[1]})
    CREATE (d)-[:TREATS]->(c)
    """
    tx.run(cypher, rels=TREATS)

def load_contraindicated(tx):
    cypher = """
    UNWIND $rels AS r
    MATCH (d:Drug {drug_id: r[0]}), (c:Condition {cond_id: r[1]})
    CREATE (d)-[:CONTRAINDICATED_IN]->(c)
    CREATE (c)-[:CONTRAINDICATES]->(d)
    """
    tx.run(cypher, rels=CONTRAINDICATED)

def load_requires_adjustment(tx):
    cypher = """
    UNWIND $rels AS r
    MATCH (d:Drug {drug_id: r[0]}), (c:Condition {cond_id: r[1]})
    CREATE (d)-[:REQUIRES_ADJUSTMENT]->(c)
    """
    tx.run(cypher, rels=REQUIRES_ADJUST)

def load_interactions(tx):
    cypher = """
    UNWIND $rows AS row
    MATCH (d1:Drug {drug_id: row[0]}), (d2:Drug {drug_id: row[1]})
    CREATE (d1)-[:INTERACTS_WITH {
        severity: row[2],
        description: row[3]
    }]->(d2)
    CREATE (d2)-[:INTERACTS_WITH {
        severity: row[2],
        description: row[3]
    }]->(d1)
    """
    tx.run(cypher, rows=INTERACTIONS)

# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------
def main():
    print(f"Loading data into database: {DATABASE_NAME}")
    with driver.session(database=DATABASE_NAME) as session:
        print("Wiping existing graph...")
        session.execute_write(wipe_db)

        print("Creating constraints...")
        session.execute_write(create_constraints)

        print("Loading drugs...")
        session.execute_write(load_drugs)

        print("Loading conditions...")
        session.execute_write(load_conditions)

        print("Loading TREATS relationships...")
        session.execute_write(load_treats)

        print("Loading CONTRAINDICATED_IN...")
        session.execute_write(load_contraindicated)

        print("Loading REQUIRES_ADJUSTMENT...")
        session.execute_write(load_requires_adjustment)

        print("Loading drug-drug INTERACTIONS...")
        session.execute_write(load_interactions)

    print("\nHard-coded KG successfully built!")
    print(f"   Database: {DATABASE_NAME}")
    print(f"   • {len(DRUGS)} Drug nodes")
    print(f"   • {len(CONDITIONS)} Condition nodes")
    print(f"   • {len(TREATS)} TREATS edges")
    print(f"   • {len(CONTRAINDICATED)} CONTRAINDICATED_IN edges")
    print(f"   • {len(REQUIRES_ADJUST)} REQUIRES_ADJUSTMENT edges")
    print(f"   • {len(INTERACTIONS)} INTERACTS_WITH edges (bidirectional)")

    driver.close()


if __name__ == "__main__":
    main()