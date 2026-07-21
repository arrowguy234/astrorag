#!/usr/bin/env python
"""Standalone environment validation script."""

import sys
from   pathlib import Path

# ensure the astrorag package is importable when script is
# executed directly from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astrorag.validate import main

if __name__ == "__main__":
    sys.exit(main())