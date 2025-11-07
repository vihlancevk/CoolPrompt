import logging
import re
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def setup_logging(logs_dir: str | Path | None = None) -> logging.Logger:
    """Logging config for CoolPrompt.

    Args:
        logs_dir: logs saving directory. Defaults to ../../logs
        relative to this file's location.
    """

    logger = logging.getLogger("coolprompt")
    if getattr(logger, "_is_configured", False):
        return logger

    if logs_dir is None:
        logs_dir = Path(__file__).parents[2] / "logs"
    Path.mkdir(logs_dir, exist_ok=True)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(module)s.%(funcName)s] - %(message)s")

    current_date = datetime.now().strftime("%Y-%m-%d")  # noqa: DTZ005
    file_handler = TimedRotatingFileHandler(
        filename=str(Path(logs_dir) / f"run_{current_date}.log"),
        when="MIDNIGHT",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d.log"
    file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler.setFormatter(formatter)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger._is_configured = True  # noqa: SLF001

    return logger


logger = setup_logging()


def set_verbose(verbose: int) -> None:
    """Sets the provided verbose level to the logger.

    Args:
        verbose (int): specifies the logging level
            0 - ERROR (only errors)
            1 - INFO (basic info + errors)
            2 - DEBUG (all messages)
    """

    logger_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}[verbose]
    logger.setLevel(logger_level)
    for handler in logger.handlers:
        handler.setLevel(logger_level)
