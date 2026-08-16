"""CLI script to export graph statistics to file (spec §2059)."""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_logging, get_settings
from src.graph.driver import close_driver, health_check
from src.ingestion.validation.validator import GraphQualityValidator

logger = logging.getLogger("scripts.export_graph")


def main() -> None:
    configure_logging()
    settings = get_settings()

    logger.info("Checking Neo4j connection...")
    hc = health_check()
    if hc.get("status") != "healthy":
        logger.error("Neo4j connection failed: %s", hc.get("error"))
        sys.exit(1)

    out_dir = settings.data_dir / "processed" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "graph_export_stats.json"

    try:
        logger.info("Exporting graph statistics...")
        validator = GraphQualityValidator()
        report = validator.generate_quality_report()

        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Graph statistics exported to %s", out_file)
    except Exception as e:
        logger.error("Failed to export graph statistics: %s", e)
        sys.exit(1)
    finally:
        close_driver()


if __name__ == "__main__":
    main()
