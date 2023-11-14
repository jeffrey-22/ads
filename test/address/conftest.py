# Configure fixtures

import pytest

@pytest.fixture
def address_module():
    # change to root dir and import needed modules
    import os, sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import fynesse.address
    return (fynesse.address)