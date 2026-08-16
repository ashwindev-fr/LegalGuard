"""CLI script to validate dataset and legal graph quality (spec §2253)."""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_logging
from src.graph.driver import close_driver
from src.ingestion.validation.validator import GraphQualityValidator

logger = logging.getLogger("scripts.validate_dataset")


def main() -> None:
    configure_logging()
    try:
        logger.info("Running dataset and graph quality checks...")
        validator = GraphQualityValidator()
        report = validator.generate_quality_report()

        print("\n" + "=" * 60)
        print("LEGAL KNOWLEDGE GRAPH QUALITY REPORT")
        print("=" * 60)
        print(f"Status: {report['status']}")
        print("\nNode Counts:")
        for label, count in report['node_counts'].items():
            print(f"  - {label}: {count}")

        print("\nQuality Checks:")
        for check, count in report['quality_checks'].items():
            print(f"  - {check}: {count}")
        print("=" * 60 + "\n")

        if report['status'] != "PASS":
            logger.warning("Quality issues detected in graph. Check report details.")
    except Exception as e:
        logger.error("Dataset validation failed: %s", e)
        sys.exit(1)
    finally:
        close_driver()


if __name__ == "__main__":
    main()
