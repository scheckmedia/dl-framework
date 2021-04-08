from dlf.core.registry import import_framework_modules
from pathlib import Path

parent = Path(__file__).parent
import_framework_modules(parent, __name__)
import_framework_modules(
    parent / "classification", __name__ + '.classification')
