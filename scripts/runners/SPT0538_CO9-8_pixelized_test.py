import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "scripts" / "bootstrap.py").is_file():
        sys.path.insert(0, str(_parent))
        break

from scripts.bootstrap import setup
from src.pipelines.runner import load_settings
from src.pipelines.runner_pixelized import run_from_settings

ROOT = setup(__file__)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settings",
        default=str(ROOT / "settings" / "runners" / "SPT0538_CO9-8_pixelized_test.json"),
    )
    args = parser.parse_args()
    run_from_settings(load_settings(args.settings))
