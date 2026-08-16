"""CLI script to recreate vector and full-text indexes (spec §2056)."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_logging
from src.graph.driver import close_driver
from src.graph.schema import create_fulltext_indexes, create_vector_index

logger = logging.getLogger("scripts.rebuild_indexes")


def main() -> None:
    configure_logging()
    try:
        logger.info("Rebuilding vector and full-text indexes...")
        create_vector_index()
        create_fulltext_indexes()
        logger.info("Index rebuild complete.")
    except Exception as e:
        logger.error("Index rebuild failed: %s", e)
        sys.exit(1)
    finally:
        close_driver()


if __name__ == "__main__":
    main()
