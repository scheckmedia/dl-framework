from dlf.core.registry import import_framework_modules
from pathlib import Path

import_framework_modules(Path(__file__).parent, __name__)
