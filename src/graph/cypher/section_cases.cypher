// Retrieve all cases that interpret, consider, or apply a given Section
MATCH (c:Case)-[r:INTERPRETS|CONSIDERS|APPLIES]->(s:Section)
WHERE s.section_number = $section_number OR s.id = $section_id
RETURN c.id AS case_id,
       c.case_name AS case_name,
       c.court AS court,
       c.judgment_date AS judgment_date,
       c.citation AS citation,
       type(r) AS relation_type,
       r.confidence AS confidence,
       r.evidence_chunk_id AS evidence_chunk_id
ORDER BY c.judgment_date DESC;
