"""Run this checkout without replacing another running job's installation."""

import os
import sys
from pathlib import Path


source = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, source)
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [source, os.environ.get("PYTHONPATH")]))

from dataflex.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
