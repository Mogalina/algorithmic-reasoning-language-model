import pytest
import shutil

from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.config import load_config
from utils.logger import setup_logger, get_logger


def test_load_config(tmp_path):
    # Create a dummy configuration file
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.yaml"
    config_file.write_text("""
    test_key: test_value
    nested:
      key: value
    """)

    # Clear the Least Recently Used (LRU) cache for testing
    load_config.cache_clear()
    
    # Resolve absolute path to the dummy config
    with patch("utils.config.PROJECT_ROOT", tmp_path):
        config = load_config("config/config.yaml")
        assert config["test_key"] == "test_value"
        assert config["nested"]["key"] == "value"


def test_get_logger():
    logger = get_logger("test_name")
    assert logger is not None


def test_setup_logger(tmp_path):
    log_file = tmp_path / "logs" / "test.log"
    setup_logger(level="DEBUG", log_file=str(log_file))
    
    # Check if the log file directory exists
    assert log_file.parent.exists()

    from loguru import logger
    logger.info("Test message for logger setup")
    
    # Check if the log file exists after message is logged
    assert log_file.exists()
