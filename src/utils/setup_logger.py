from loguru import logger
from pathlib import Path
from rich.logging import RichHandler
import shutil
from typing import Optional, Union

def setup_logger(
    log_level: str = "INFO",
    log_folder: Optional[Union[Path, str]] = None,
    file_log_level: Optional[str] = None,
    recreate_folder: bool = True
) -> None:
    """
    Setup logger for benchmark with Rich console output and file logging.
    
    Args:
        log_level: Console logging level
        log_folder: Path to folder for log files. If None, file logging is disabled.
        file_log_level: File logging level. If None, uses same level as console.
        recreate_folder: If True, completely recreates the log folder
    """

    logger.remove()
    
    logger.add(
        RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_level=True,
            show_path=True,
        ),
        level=log_level,
    )
    
    if log_folder:
        log_folder = Path(log_folder).resolve()
        
        if recreate_folder and log_folder.exists():
            shutil.rmtree(log_folder)

        log_folder.mkdir(parents=True, exist_ok=True)
        
        file_level = file_log_level or log_level
        
        logger.add(
            log_folder / "general.log",
            level=file_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {function}:{line} - {message}",
            rotation="100 MB",
            retention="30 days",
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
