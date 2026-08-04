import sys
from pathlib import Path

#------------------------------------------------------------------------------
#Example:
#  python scripts/run_pixelized_fit.py --settings settings/runners/SPT0538_CO9-8_pixelized.json (recommended)
#  python scripts/runners/SPT0538_CO9-8_pixelized.py
#------------------------------------------------------------------------------

for _parent in Path(__file__).resolve().parents:
    if (_parent / "scripts" / "bootstrap.py").is_file():
        sys.path.insert(0, str(_parent))
        break

from scripts.bootstrap import setup
from src.pipelines.runner_pixelized import main

ROOT = setup(__file__)

if __name__ == "__main__":
    main(default_settings_path=ROOT / "settings" / "runners" / "SPT0538_CO9-8_pixelized.json")
