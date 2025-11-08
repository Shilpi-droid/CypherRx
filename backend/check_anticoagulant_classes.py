#!/usr/bin/env python3
"""Check anticoagulant drug classes in the database"""

from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'neo4j_password'))
)

with driver.session(database=os.getenv('DATABASE_NAME', 'neo4j')) as session:
    # Check anticoagulant drugs
    result = session.run("""
        MATCH (d:Drug)
        WHERE d.name IN ['Warfarin', 'Apixaban', 'Rivaroxaban', 'Dabigatran', 'Edoxaban', 'Enoxaparin']
        RETURN d.name AS name, d.class AS class
        ORDER BY d.name
    """)
    
    print("Anticoagulant drugs and their classes:")
    for record in result:
        print(f"  {record['name']}: {record['class']}")
    
    # Check which ones treat Atrial Fibrillation
    result2 = session.run("""
        MATCH (d:Drug)-[:TREATS]->(c:Condition {name: 'Atrial Fibrillation'})
        WHERE d.name IN ['Warfarin', 'Apixaban', 'Rivaroxaban', 'Dabigatran', 'Edoxaban', 'Enoxaparin']
        RETURN d.name AS name, d.class AS class
        ORDER BY d.name
    """)
    
    print("\nAnticoagulants that TREAT Atrial Fibrillation:")
    for record in result2:
        print(f"  {record['name']}: {record['class']}")
    
    # Check which ones are contraindicated in CKD
    result3 = session.run("""
        MATCH (d:Drug)-[:CONTRAINDICATED_IN]->(c:Condition {name: 'Chronic Kidney Disease'})
        WHERE d.name IN ['Warfarin', 'Apixaban', 'Rivaroxaban', 'Dabigatran', 'Edoxaban', 'Enoxaparin']
        RETURN d.name AS name, d.class AS class
        ORDER BY d.name
    """)
    
    print("\nAnticoagulants CONTRAINDICATED in Chronic Kidney Disease:")
    for record in result3:
        print(f"  {record['name']}: {record['class']}")
    
    # Check which ones require adjustment in CKD
    result4 = session.run("""
        MATCH (d:Drug)-[:REQUIRES_ADJUSTMENT]->(c:Condition {name: 'Chronic Kidney Disease'})
        WHERE d.name IN ['Warfarin', 'Apixaban', 'Rivaroxaban', 'Dabigatran', 'Edoxaban', 'Enoxaparin']
        RETURN d.name AS name, d.class AS class
        ORDER BY d.name
    """)
    
    print("\nAnticoagulants requiring ADJUSTMENT in Chronic Kidney Disease:")
    for record in result4:
        print(f"  {record['name']}: {record['class']}")

driver.close()

