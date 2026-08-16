"""CLI script to run evaluation benchmark (spec §3748)."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_logging
from src.evaluation.runner import BenchmarkRunner
from src.graph.driver import close_driver

logger = logging.getLogger("scripts.run_evaluation")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run legal system evaluation benchmark")
    parser.add_argument(
        "--system",
        choices=["all", "baseline", "vector_rag", "graphrag", "finetuned_graphrag"],
        default="all",
        help="System variant to evaluate",
    )
    args = parser.parse_args()

    try:
        runner = BenchmarkRunner()

        if args.system == "all":
            logger.info("Running evaluation across all 4 system baselines...")
            table_md = runner.generate_comparison_table()
            print("\n" + "=" * 60)
            print("EVALUATION BENCHMARK RESULTS")
            print("=" * 60)
            print(table_md)
            print("=" * 60 + "\n")
        else:
            metrics = runner.evaluate_system(args.system)
            print(f"\nResults for {args.system}:")
            print(json.dumps(metrics.__dict__, indent=2))
    except Exception as e:
        logger.error("Evaluation run failed: %s", e)
        sys.exit(1)
    finally:
        close_driver()


if __name__ == "__main__":
    main()
