"""CLI script for batch document ingestion (spec §2235). Supports --dry-run."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_logging
from src.graph.driver import close_driver
from src.ingestion.models import DocumentType
from src.ingestion.sources.manual_import import ingest_directory

logger = logging.getLogger("scripts.ingest_batch")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Batch ingest legal documents from directory")
    parser.add_argument("--dir", required=True, help="Directory containing PDF/text files")
    parser.add_argument("--type", default="ACT", help="Document type for batch")
    parser.add_argument("--source-id", default="MANUAL", help="Source ID")
    parser.add_argument("--authority", type=int, default=5, help="Authority level (0-5)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate batch ingestion without writing to graph")
    args = parser.parse_args()

    try:
        doc_type = DocumentType(args.type.upper())
    except ValueError:
        logger.error("Invalid document type: %s", args.type)
        sys.exit(1)

    try:
        results = ingest_directory(
            directory=args.dir,
            document_type=doc_type,
            source_id=args.source_id,
            authority_level=args.authority,
            dry_run=args.dry_run,
        )
        logger.info("Batch ingestion summary: %d files processed", len(results))
    except Exception as e:
        logger.error("Batch ingestion failed: %s", e)
        sys.exit(1)
    finally:
        close_driver()


if __name__ == "__main__":
    main()
