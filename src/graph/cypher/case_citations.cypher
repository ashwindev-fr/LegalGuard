// Retrieve citation graph for a case: cases cited by it, cases following/distinguishing/overruling it
MATCH (c:Case {id: $case_id})
OPTIONAL MATCH (c)-[r1:CITES|FOLLOWS|DISTINGUISHES|OVERRULES]->(target:Case)
OPTIONAL MATCH (source:Case)-[r2:CITES|FOLLOWS|DISTINGUISHES|OVERRULES]->(c)
RETURN c.id AS case_id,
       c.case_name AS case_name,
       collect(DISTINCT {
         target_id: target.id,
         target_name: target.case_name,
         relation: type(r1),
         evidence: r1.evidence_chunk_id
       }) AS outgoing_citations,
       collect(DISTINCT {
         source_id: source.id,
         source_name: source.case_name,
         relation: type(r2),
         evidence: r2.evidence_chunk_id
       }) AS incoming_citations;
