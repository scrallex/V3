from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_PATH = PROJECT_ROOT / "scripts"
INIT_FILE = SCRIPTS_PATH / "__init__.py"
if INIT_FILE.exists():
    need_reload = True
    existing = sys.modules.get("scripts")
    if existing is not None:
        module_path = Path(getattr(existing, "__file__", "")).resolve()
        if module_path.parent == SCRIPTS_PATH:
            need_reload = False
    if need_reload:
        spec = importlib.util.spec_from_file_location(
            "scripts",
            INIT_FILE,
            submodule_search_locations=[str(SCRIPTS_PATH)],
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            module.__path__ = [str(SCRIPTS_PATH)]  # type: ignore[attr-defined]
            sys.modules["scripts"] = module
            spec.loader.exec_module(module)
