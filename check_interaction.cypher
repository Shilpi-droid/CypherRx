// Diagnostic queries to check Warfarin-Ibuprofen interaction

// 1. Check if Ibuprofen exists in database
MATCH (d:Drug {name: "Ibuprofen"})
RETURN d.name AS name, d.drug_id AS drug_id, d.class AS class;

// 2. Check if Warfarin exists in database
MATCH (d:Drug {name: "Warfarin"})
RETURN d.name AS name, d.drug_id AS drug_id, d.class AS class;

// 3. Check interaction between Warfarin and Ibuprofen (bidirectional)
MATCH (d1:Drug {name: "Warfarin"})-[r:INTERACTS_WITH]-(d2:Drug {name: "Ibuprofen"})
RETURN d1.name AS drug1, 
       d2.name AS drug2,
       r.severity AS severity,
       r.description AS description;

// 4. Check all drugs with drug_id D061 (should only be Ibuprofen)
MATCH (d:Drug)
WHERE d.drug_id = "D061"
RETURN d.name AS name, d.drug_id AS drug_id, d.class AS class;

// 5. Check all interactions for Warfarin
MATCH (d1:Drug {name: "Warfarin"})-[r:INTERACTS_WITH]-(d2:Drug)
RETURN d1.name AS drug1, 
       d2.name AS drug2,
       d2.drug_id AS drug2_id,
       r.severity AS severity,
       r.description AS description
ORDER BY r.severity DESC;

// 6. Check all drugs with name containing "Ibu" or "Nitro"
MATCH (d:Drug)
WHERE d.name CONTAINS "Ibu" OR d.name CONTAINS "Nitro"
RETURN d.name AS name, d.drug_id AS drug_id, d.class AS class;

