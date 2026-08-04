# Run from the LensKin repo directory, e.g.:
#  python scripts/make_cornerplot.py /path/to/run_directory
#  python scripts/make_cornerplot.py output/SPT0538_CO9-8/phase_mass[1]_m3_m4/7b6baaafded32d847303b60fe9695f8c

import argparse
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "scripts" / "bootstrap.py").is_file():
        sys.path.insert(0, str(_parent))
        break

from scripts.bootstrap import setup
from src.pipelines.cornerplot import make_cornerplot

setup(__file__)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Make a corner plot of free parameters from a completed "
            "PyAutoFit run (files/samples.csv)."
        )
    )
    parser.add_argument(
        "run_directory",
        help="PyAutoFit run directory (hash folder) or path to files/samples.csv",
    )
    args = parser.parse_args()

    output_path = make_cornerplot(run_directory=args.run_directory)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
