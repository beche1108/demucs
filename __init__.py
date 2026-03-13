from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_INNER_PACKAGE_ROOT = _PACKAGE_ROOT / "demucs"
_INNER_INIT = _INNER_PACKAGE_ROOT / "__init__.py"

if not _INNER_INIT.is_file():
    raise ImportError(f"Demucs inner package is missing: {_INNER_INIT}")

# When the workspace root is on sys.path, Python would otherwise treat the
# submodule checkout directory as a namespace package. That makes
# `import demucs...` resolve inconsistently between the editable install and the
# local checkout, and breaks relative imports such as `.demucs` inside the
# upstream package. Force the package search path to the real upstream package
# directory so both code paths behave identically.
__path__ = [str(_INNER_PACKAGE_ROOT)]

exec(compile(_INNER_INIT.read_text(encoding="utf-8"), str(_INNER_INIT), "exec"))
