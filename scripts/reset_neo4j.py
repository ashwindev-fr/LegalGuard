"""CLI script to reset Neo4j database (spec §100). Requires explicit --confirm flag."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_logging
from src.graph.driver import close_driver, get_session

logger = logging.getLogger("scripts.reset_neo4j")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Reset Neo4j database (DESTRUCTIVE)")
    parser.add_argument("--confirm", action="store_true", help="Confirm database reset")
    args = parser.parse_args()

    if not args.confirm:
        logger.error("Database reset aborted! You must pass the '--confirm' flag.")
        sys.exit(1)

    try:
        logger.warning("Clearing all nodes and relationships in Neo4j...")
        query = "MATCH (n) DETACH DELETE n"
        with get_session() as session:
            session.run(query)
        logger.info("Database reset complete.")
    except Exception as e:
        logger.error("Database reset failed: %s", e)
        sys.exit(1)
    finally:
        close_driver()


if __name__ == "__main__":
    main()
