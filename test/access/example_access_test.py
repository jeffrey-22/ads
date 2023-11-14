# Change directory to root for importing the fynesse module
import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import fynesse.access

def func(x):
    return x + 2

def test_answer():
    assert func(3) == 5

def test_imported():
    assert fynesse.access.add_one(4) == 5