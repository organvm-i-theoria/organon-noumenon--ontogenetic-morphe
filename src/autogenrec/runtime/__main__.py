"""
Entry point for running the AutoGenRec system.

Usage:
    python -m autogenrec.runtime
"""

import asyncio
import signal
import sys
from typing import NoReturn

import structlog

from autogenrec.runtime.orchestrator import create_default_orchestrator

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


async def run_system() -> None:
    """Run the AutoGenRec system."""
    logger.info("autogenrec_initializing")

    orchestrator = create_default_orchestrator()

    # Set up signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def handle_signal(sig: signal.Signals) -> None:
        logger.info("shutdown_signal_received", signal=sig.name)
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal, sig)

    try:
        async with orchestrator.run_context():
            logger.info(
                "autogenrec_running",
                subsystems=orchestrator.subsystem_registry.list_names(),
            )

            # Print health status
            health = orchestrator.get_health()
            logger.info("system_health", **health)

            # Wait for shutdown signal
            await shutdown_event.wait()

    except Exception:
        logger.exception("autogenrec_error")
        raise

    logger.info("autogenrec_stopped")


def main() -> NoReturn:
    """Main entry point."""
    try:
        asyncio.run(run_system())
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("interrupted")
        sys.exit(130)
    except Exception:
        logger.exception("fatal_error")
        sys.exit(1)


if __name__ == "__main__":
    main()
