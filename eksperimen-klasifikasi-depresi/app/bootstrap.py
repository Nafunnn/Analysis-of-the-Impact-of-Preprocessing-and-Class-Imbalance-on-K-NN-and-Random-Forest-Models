"""Bootstrap path agar `utils` bisa di-import dari Home maupun pages."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_app_on_path() -> Path:
    """Pastikan direktori `app/` ada di sys.path; kembalikan path tersebut."""
    here = Path(__file__).resolve().parent
    # Dipanggil dari Home.py → parent = app; dari pages/*.py → parent.parent = app
    app_dir = here if (here / "utils").is_dir() else here.parent
    app_str = str(app_dir)
    if app_str not in sys.path:
        sys.path.insert(0, app_str)
    return app_dir
