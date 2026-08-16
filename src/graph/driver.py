"""Neo4j driver management — singleton connection with health checking."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

from neo4j import Driver, GraphDatabase, Session

from src.config import get_settings

logger = logging.getLogger(__name__)

_driver: Driver | None = None


def get_driver() -> Driver:
    """Return (and cache) a Neo4j driver instance."""
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        logger.info("Neo4j driver created → %s", settings.neo4j_uri)
    return _driver


def close_driver() -> None:
    """Close the cached driver."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
        logger.info("Neo4j driver closed")


@contextmanager
def get_session(database: str | None = None) -> Generator[Session, None, None]:
    """Yield a Neo4j session (auto-closed)."""
    settings = get_settings()
    db = database or settings.neo4j_database
    driver = get_driver()
    session = driver.session(database=db)
    try:
        yield session
    finally:
        session.close()


def run_query(
    query: str,
    parameters: dict[str, Any] | None = None,
    database: str | None = None,
) -> list[dict[str, Any]]:
    """Execute a Cypher query and return all records as dicts."""
    with get_session(database) as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]


def health_check() -> dict[str, Any]:
    """Return Neo4j connectivity status."""
    try:
        driver = get_driver()
        driver.verify_connectivity()
        # Get server info
        with get_session() as session:
            result = session.run("CALL dbms.components() YIELD name, versions")
            record = result.single()
            return {
                "status": "healthy",
                "name": record["name"] if record else "unknown",
                "versions": record["versions"] if record else [],
            }
    except Exception as e:
        logger.error("Neo4j health check failed: %s", e)
        return {"status": "unhealthy", "error": str(e)}
