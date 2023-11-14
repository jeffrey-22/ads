# Configure fixtures

import pytest

@pytest.fixture
def assess_module():
    # change to root dir and import needed modules
    import os, sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import fynesse.assess
    return (fynesse.assess)