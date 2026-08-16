// Retrieve cases decided by a specific court within optional date bounds
MATCH (c:Case)-[:DECIDED_BY]->(court:Court {name: $court_name})
WHERE ($start_date IS NULL OR c.judgment_date >= $start_date)
  AND ($end_date IS NULL OR c.judgment_date <= $end_date)
RETURN c.id AS case_id,
       c.case_name AS case_name,
       c.judgment_date AS judgment_date,
       c.citation AS citation
ORDER BY c.judgment_date DESC
LIMIT $limit;
