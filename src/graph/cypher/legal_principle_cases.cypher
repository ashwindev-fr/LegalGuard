// Retrieve legal principles established or supported by cases
MATCH (c:Case)-[:ESTABLISHES]->(p:LegalPrinciple)
OPTIONAL MATCH (chunk:Chunk)-[:SUPPORTS]->(p)
RETURN p.id AS principle_id,
       p.title AS principle_title,
       p.description AS description,
       c.id AS case_id,
       c.case_name AS case_name,
       collect(DISTINCT chunk.id) AS supporting_chunk_ids;
