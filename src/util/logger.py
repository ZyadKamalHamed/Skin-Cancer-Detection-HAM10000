import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_logger(
        name: str,
        level: str =  "INFO",
        log_file: Optional[str] = None,
        format_string: Optional[str] = None)-> logging.Logger:
    """
    set up a logger with colored console output and optional file logging.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file
        format_string: Optional custom format string.

        Returns:
            Configured logger instance,
    """

    #Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    # Default format with colors using ANSI codes
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        #create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setlevel(getattr(logging, level.upper()))

    #Create formatter
    class ColoredFormatter(logging.Formatter):
        """Custom Formatter with colors for different log levels"""

        #ANSI colour codes
        COLORS = {
            'DEBUG': '\033[36m',  #CYAN
            'INFO': '\033[32m',   #GREEN
            'WARNING': '\033[33m',#YELLOW
            'ERROR': '\033[31m',  #RED
            'CRITICAL': '\033[35m',#MAGENTA
            'RESET': '\033[0m'     #RESET
        }

        def format(self, record):
            #Add collour to Levelname
            levelname_color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{levelname_color}{record.levelname}{self.COLORS['RESET']}"
            return super().format(record)

        #Set the formatter
        formater = ColoredFormatter(format_string)
        console_handler.setFormatter(formater)
        logger.addHandler(console_handler)

        #File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_path.DEBUG) #File gets all messages
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcNames)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        return logger

        def get_project_logger(name: str) -> logging.Logger:
            """
            Get a logger configured for the project.

            Args:
                name: Logger name (usually __name__)

            :return:
                Configured logger for the project.
            """

            #Create logs directory
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)

            #Create log file with timestamp
            timestap = datetime.now().strftime("%Y%m%d")
            log_file = logs_dir / f"skin_cancer_detection_{timestamp}.log"

            return setup_logger(
                name = name,
                level="INFO",
                log_file=str(log_file)
            )
        #Create a default project loader
        project_logger = get_project_logger("skin_cancer_detection")

        if __name__ == "__main__":
            #Test the logger
            logger = get_project_logger(__name__)

            logger.debug("This is a debug message")
            logger.info("This is an info message")
            logger.warning("This is a warning message")
            logger.error("This is an error message")
            logger.critical("This is a critical message")

            print("\n Logger test completed, check logs/ directory for output of log file")



