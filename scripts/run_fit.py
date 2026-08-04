import sys
from pathlib import Path

#------------------------------------------------------------------------------
# Run from the LensKin repo directory, e.g.:
#  python scripts/run_fit.py --settings settings/runners/SPT0538_CO9-8.json
#  python scripts/run_fit.py --settings settings/runners/SPT0538_CO9-8_pixelized.json
#  python scripts/runners/HERMES_J021830.5-053124.py
#------------------------------------------------------------------------------

for _parent in Path(__file__).resolve().parents:
    if (_parent / "scripts" / "bootstrap.py").is_file():
        sys.path.insert(0, str(_parent))
        break

import argparse

from scripts.bootstrap import setup
from src.pipelines.lens_model import validate_lensing_settings
from src.pipelines.normalization import requires_phase1, validate_normalization_settings
from src.pipelines.runner import load_settings, run_from_settings as run_single_phase
from src.pipelines.runner_pixelized import run_from_settings as run_two_phase

setup(__file__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a LensKin fit from a settings JSON file.")
    parser.add_argument(
        "--settings",
        required=True,
        help="Path to runner settings JSON (e.g. settings/runners/SPT0538_CO9-8.json)",
    )
    args = parser.parse_args()
    settings = load_settings(args.settings)
    validate_lensing_settings(settings)
    mode = validate_normalization_settings(settings)
    if requires_phase1(mode):
        run_two_phase(settings)
    else:
        run_single_phase(settings)
