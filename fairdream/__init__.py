import logging

from fairdream.utils.logger_config import setup_log_config

__version__ = "0.0.1"

setup_log_config(is_dev=False)
LOGGER = logging.getLogger(__name__)
