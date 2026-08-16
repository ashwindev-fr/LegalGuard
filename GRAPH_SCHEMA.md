# Neo4j Graph Schema Specification

## Node Labels

- `Act`: Legislation document (`id`, `title`, `short_title`, `year`, `status`)
- `Article`: Constitutional article (`id`, `article_number`, `title`)
- `Section`: Statutory provision (`id`, `section_number`, `title`, `effective_from`, `effective_until`)
- `Case`: Judicial decision (`id`, `case_name`, `court`, `judgment_date`, `citation`)
- `Court`: Judicial body (`id`, `name`)
- `Judge`: Presiding judge (`id`, `name`)
- `Chunk`: Text chunk (`id`, `text`, `embedding`, `page_start`, `paragraph_start`)
- `Source`: Provenance origin (`id`, `name`, `base_url`, `authority_level`)
- `LegalPrinciple`: Judicial ratio/principle (`id`, `title`, `description`)
- `Amendment`: Legislative amendment (`id`, `title`, `year`, `effective_date`)

## Relationships

- `(:Act)-[:CONTAINS]->(:Section)`
- `(:Section)-[:HAS_SUBSECTION]->(:SubSection)`
- `(:Case)-[:CONSIDERS|INTERPRETS|APPLIES]->(:Section)`
- `(:Case)-[:CITES|FOLLOWS|DISTINGUISHES|OVERRULES]->(:Case)`
- `(:Case)-[:DECIDED_BY]->(:Court)`
- `(:Case)-[:ESTABLISHES]->(:LegalPrinciple)`
- `(:Document)-[:HAS_CHUNK]->(:Chunk)`
- `(:Chunk)-[:FROM_SOURCE]->(:Source)`
- `(:Section)-[:AMENDED_BY|REPEALED_BY]->(:Amendment)`
