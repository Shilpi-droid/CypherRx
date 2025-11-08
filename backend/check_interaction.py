#!/usr/bin/env python3
"""
Quick diagnostic script to check if Warfarin-Ibuprofen interaction exists in database
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Neo4j connection
uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("DATABASE_NAME", "neo4j")

driver = GraphDatabase.driver(uri, auth=(user, password))

print("=" * 70)
print("DIAGNOSTIC: Checking Warfarin-Ibuprofen Interaction")
print("=" * 70)

with driver.session(database=database) as session:
    # 1. Check if Ibuprofen exists
    print("\n1. Checking if Ibuprofen exists in database:")
    result = session.run("MATCH (d:Drug {name: 'Ibuprofen'}) RETURN d.name AS name, d.drug_id AS drug_id, d.class AS class")
    ibuprofen = list(result)
    if ibuprofen:
        for record in ibuprofen:
            print(f"   ✓ Found: {record['name']} (ID: {record['drug_id']}, Class: {record['class']})")
    else:
        print("   ✗ Ibuprofen NOT FOUND in database!")
    
    # 2. Check if Warfarin exists
    print("\n2. Checking if Warfarin exists in database:")
    result = session.run("MATCH (d:Drug {name: 'Warfarin'}) RETURN d.name AS name, d.drug_id AS drug_id, d.class AS class")
    warfarin = list(result)
    if warfarin:
        for record in warfarin:
            print(f"   ✓ Found: {record['name']} (ID: {record['drug_id']}, Class: {record['class']})")
    else:
        print("   ✗ Warfarin NOT FOUND in database!")
    
    # 3. Check interaction (bidirectional)
    print("\n3. Checking interaction between Warfarin and Ibuprofen (bidirectional):")
    result = session.run("""
        MATCH (d1:Drug {name: 'Warfarin'})-[r:INTERACTS_WITH]-(d2:Drug {name: 'Ibuprofen'})
        RETURN d1.name AS drug1, 
               d2.name AS drug2,
               r.severity AS severity,
               r.description AS description
    """)
    interaction = list(result)
    if interaction:
        for record in interaction:
            print(f"   ✓ INTERACTION FOUND:")
            print(f"     {record['drug1']} <-> {record['drug2']}")
            print(f"     Severity: {record['severity']}")
            print(f"     Description: {record['description']}")
    else:
        print("   ✗ NO INTERACTION FOUND between Warfarin and Ibuprofen!")
    
    # 4. Check all drugs with drug_id D061 (should only be Ibuprofen)
    print("\n4. Checking all drugs with drug_id 'D061':")
    result = session.run("MATCH (d:Drug) WHERE d.drug_id = 'D061' RETURN d.name AS name, d.drug_id AS drug_id, d.class AS class")
    d061_drugs = list(result)
    if d061_drugs:
        print(f"   Found {len(d061_drugs)} drug(s) with ID D061:")
        for record in d061_drugs:
            print(f"     - {record['name']} (Class: {record['class']})")
        if len(d061_drugs) > 1:
            print("   ⚠ WARNING: Multiple drugs with same ID! This is a problem.")
    else:
        print("   ✗ No drug found with ID D061!")
    
    # 5. Check all interactions for Warfarin
    print("\n5. All interactions for Warfarin:")
    result = session.run("""
        MATCH (d1:Drug {name: 'Warfarin'})-[r:INTERACTS_WITH]-(d2:Drug)
        RETURN d1.name AS drug1, 
               d2.name AS drug2,
               d2.drug_id AS drug2_id,
               r.severity AS severity,
               r.description AS description
        ORDER BY r.severity DESC
    """)
    all_interactions = list(result)
    if all_interactions:
        print(f"   Found {len(all_interactions)} interaction(s):")
        for record in all_interactions:
            print(f"     - {record['drug1']} <-> {record['drug2']} (ID: {record['drug2_id']})")
            print(f"       Severity: {record['severity']}, Description: {record['description']}")
    else:
        print("   ✗ No interactions found for Warfarin!")
    
    # 6. Check for drugs with similar names
    print("\n6. Checking for drugs with 'Ibu' or 'Nitro' in name:")
    result = session.run("""
        MATCH (d:Drug)
        WHERE d.name CONTAINS 'Ibu' OR d.name CONTAINS 'Nitro'
        RETURN d.name AS name, d.drug_id AS drug_id, d.class AS class
        ORDER BY d.name
    """)
    similar_drugs = list(result)
    if similar_drugs:
        for record in similar_drugs:
            print(f"     - {record['name']} (ID: {record['drug_id']}, Class: {record['class']})")
    else:
        print("     No drugs found with 'Ibu' or 'Nitro' in name")

print("\n" + "=" * 70)
print("Diagnostic complete!")
print("=" * 70)

driver.close()

