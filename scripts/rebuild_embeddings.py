"""CLI script to re-embed all chunks and update Neo4j (spec §3303)."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_logging
from src.embeddings import embed_all_chunks
from src.graph.driver import close_driver

logger = logging.getLogger("scripts.rebuild_embeddings")


def main() -> None:
    configure_logging()
    try:
        logger.info("Starting embedding generation for un-embedded chunks...")
        count = embed_all_chunks()
        logger.info("Finished embedding generation: %d chunks updated.", count)
    except Exception as e:
        logger.error("Embedding generation failed: %s", e)
        sys.exit(1)
    finally:
        close_driver()


if __name__ == "__main__":
    main()
