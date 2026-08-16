"""CLI script to initialize Neo4j constraints and indexes (spec §2210)."""

import logging
import sys
from pathlib import Path

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_logging
from src.graph.driver import close_driver, health_check
from src.graph.schema import init_schema

logger = logging.getLogger("scripts.init_neo4j")


def main() -> None:
    configure_logging()
    logger.info("Checking Neo4j connection...")
    try:
        hc = health_check()
        if hc.get("status") != "healthy":
            logger.error("Neo4j connection failed: %s", hc.get("error"))
            sys.exit(1)

        logger.info("Neo4j connection verified (%s). Initializing graph schema...", hc.get("versions"))
        init_schema()
        logger.info("Graph schema initialization complete.")
    except Exception as e:
        logger.error("Failed to initialize Neo4j schema: %s", e)
        sys.exit(1)
    finally:
        close_driver()


if __name__ == "__main__":
    main()
