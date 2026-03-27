import pytest
import sys
import os
from pathlib import Path


# Add project root and source code to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))


# Set environment variables for testing
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
