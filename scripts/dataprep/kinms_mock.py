import sys
from pathlib import Path

try:
    _SCRIPT_FILE = __file__
except NameError:
    _SCRIPT_FILE = None

if _SCRIPT_FILE is not None:
    _search_paths = Path(_SCRIPT_FILE).resolve().parents
else:
    _search_paths = [Path.cwd(), *Path.cwd().parents]

for _parent in _search_paths:
    if (_parent / "scripts" / "bootstrap.py").is_file():
        _repo_str = str(_parent)
        if _repo_str not in sys.path:
            sys.path.insert(0, _repo_str)
        break

from scripts.bootstrap import bootstrap_script
from src.pipelines.dataprep import main
from src.pipelines.dataprep_lib import *  # noqa: F401,F403

ROOT = bootstrap_script(_SCRIPT_FILE)

if __name__ == "__main__" or _SCRIPT_FILE is None:
    main(
        module_globals=globals(),
        default_settings_path=str(ROOT / "settings" / "dataprep" / "kinms_mock.json"),
        repo_root=ROOT,
    )
