"""
Configures a centralized logger for the project.
"""
import logging
import sys
from pathlib import Path

def setup_logging(log_file_name: str = "project.log"):
    """
    Configures logging to both file and console.

    Args:
        log_file_name (str): The name of the log file to create in the /logs directory.
    
    Returns:
        logging.Logger: The configured root logger instance.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / log_file_name

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)-15s : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[]
    )
    
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())                  # Print to console
    logger.addHandler(logging.FileHandler(log_file, mode='a'))  # Append to the log file
  
    logger.info("="*50)
    logger.info(f"Logging configured. Log file: {log_file.resolve()}")
    logger.info("="*50)
    return logger
