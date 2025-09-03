import logging
import logging.config
from pathlib import Path

import yaml

from .core.orchestration import Orchestrator


def setup_logging() -> None:
    """
    Configure the logging system for the application.

    Loads the logging configuration from a YAML file, ensures the `logs/` directory exists,
    and applies the configuration globally.

    Raises:
        FileNotFoundError: If the logger configuration file is missing.
        ValueError: If the logging configuration is invalid.
    """
    config_path = Path("config/logger_config.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Create logs directory if it does not exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Apply configuration
    logging.config.dictConfig(config)

    logger = logging.getLogger("app")
    logger.info("Logging system successfully configured")


def main() -> int:
    """
    Main entrypoint for LiteLLM Studio.

    Returns:
        int: Exit code (0 = success, 1 = failure).
    """
    # Configure logging
    setup_logging()

    # Get main logger
    logger = logging.getLogger("app")

    logger.info("Starting LiteLLM Studio")

    try:
        # Create orchestrator instance
        orchestrator = Orchestrator()

        # Run full pipeline
        success = orchestrator.run_full_pipeline()

        if success:
            logger.info("Execution completed successfully")

            # Run scanner for detailed hardware report
            hardware_report = orchestrator.hardware_scanner.scan()

            # Save report to JSON
            json_report = hardware_report.to_json()
            with open("hardware_report.json", "w", encoding="utf-8") as f:
                f.write(json_report)

            logger.info("Hardware report saved to hardware_report.json")

            return 0
        else:
            logger.error("Pipeline execution failed")
            return 1

    except Exception as e:
        logger.error(f"Critical error during execution: {str(e)}", exc_info=True)
        print(f"Error running LiteLLM Studio: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
