// Retrieve legislative amendment history for an Act or Section
MATCH (s:Section {id: $section_id})-[r:AMENDED_BY|REPEALED_BY|REPLACED_BY]->(a:Amendment)
RETURN s.id AS section_id,
       s.section_number AS section_number,
       type(r) AS amendment_relation,
       a.id AS amendment_id,
       a.title AS amendment_title,
       a.year AS amendment_year,
       a.effective_date AS effective_date
ORDER BY a.year ASC;
