// Retrieve section details applicable to a target date
MATCH (s:Section)
WHERE (s.section_number = $section_number OR s.id = $section_id)
  AND (s.effective_from IS NULL OR s.effective_from <= $target_date)
  AND (s.effective_until IS NULL OR s.effective_until >= $target_date)
RETURN s.id AS section_id,
       s.title AS title,
       s.text AS text,
       s.effective_from AS effective_from,
       s.effective_until AS effective_until,
       s.status AS status;
