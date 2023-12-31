# Configure fixtures

import pytest
import os, sys
from pathlib import Path

@pytest.fixture
def assess_module():
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import fynesse.assess
    return (fynesse.assess)