"""CLI script to ingest a single document (spec §2231)."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_logging
from src.graph.driver import close_driver
from src.ingestion.models import DocumentType
from src.ingestion.sources.manual_import import ManualImporter

logger = logging.getLogger("scripts.ingest_document")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Ingest a single legal document")
    parser.add_argument("--path", required=True, help="Path to PDF or text document")
    parser.add_argument("--type", default="ACT", help="Document type (ACT, CONSTITUTION, JUDGMENT, etc.)")
    parser.add_argument("--title", help="Document title")
    parser.add_argument("--source-id", default="MANUAL", help="Source ID")
    parser.add_argument("--authority", type=int, default=5, help="Authority level (0-5)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate ingestion without writing to graph")
    args = parser.parse_args()

    doc_path = Path(args.path)
    title = args.title or doc_path.stem.replace("_", " ").title()

    try:
        doc_type = DocumentType(args.type.upper())
    except ValueError:
        logger.error("Invalid document type: %s", args.type)
        sys.exit(1)

    try:
        importer = ManualImporter()
        result = importer.ingest_file(
            file_path=doc_path,
            document_type=doc_type,
            title=title,
            source_id=args.source_id,
            authority_level=args.authority,
            dry_run=args.dry_run,
        )
        logger.info("Ingestion result: %s", result)
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        sys.exit(1)
    finally:
        close_driver()


if __name__ == "__main__":
    main()
