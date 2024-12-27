# logger.py
import logging
import traceback
from datetime import datetime
import os

class CustomLogger:
    def __init__(self):
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Set up logging format
        self.logger = logging.getLogger('HAP')
        self.logger.setLevel(logging.DEBUG)

        # Create handlers for different log levels
        self._setup_handlers()

    def _setup_handlers(self):
        # Generate timestamp for log files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Info log handler
        info_handler = logging.FileHandler(f'logs/info_{timestamp}.log')
        info_handler.setLevel(logging.INFO)
        info_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        info_handler.setFormatter(info_formatter)

        # Error log handler
        error_handler = logging.FileHandler(f'logs/error_{timestamp}.log')
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n')
        error_handler.setFormatter(error_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(info_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)

    def log_message(self, message: str):
        """Log informational messages"""
        self.logger.info(message)

    def log_exception(self, error_message: str, exc_info=None):
        """Log exceptions with stack trace"""
        if exc_info:
            self.logger.error(f"{error_message}\n{traceback.format_exc()}", exc_info=True)
        else:
            self.logger.error(error_message)

    def log_debug(self, message: str):
        """Log debug messages"""
        self.logger.debug(message)

    def log_warning(self, message: str):
        """Log warning messages"""
        self.logger.warning(message)

# Create a singleton instance
logger = CustomLogger()

# Export convenient functions
log_message = logger.log_message
log_exception = logger.log_exception
log_debug = logger.log_debug
log_warning = logger.log_warning