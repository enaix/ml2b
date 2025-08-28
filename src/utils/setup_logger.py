from loguru import logger
from pathlib import Path
from rich.logging import RichHandler

def setup_logger(log_level: int, log_folder: Path | None | str = None) -> None:
    """
    Setup logger for benchmark

    Args:
        log_level (int): logging level
        log_path (Path | None | str, optional): Path to log file if needed. Defaults to None.
    """
    logger.remove()
    if log_folder:
        log_folder = Path(log_folder).resolve()
        log_folder.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_folder / "bench.log",
            level=log_level,
            format="[{time:YYYY-MM-DD HH:mm:ss}] >> {level} >> {message}",
            rotation="10 MB",
            encoding="utf-8",
            enqueue=True 
        )

    logger.add(
        RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_level=True,
        ),
        level=log_level,
    )
