"""Project-scoped Python startup tweaks.

Only activates when `WTA_EXTRA_SITE_PACKAGES` is set by a launcher. This lets
us append external site-packages (for example the vLLM environment) after the
active virtualenv so the active env keeps priority for shared dependencies.
"""

from __future__ import annotations

import os
import site


def _append_extra_site_packages() -> None:
    raw_paths = os.environ.get("WTA_EXTRA_SITE_PACKAGES", "")
    if not raw_paths:
        return
    for path in raw_paths.split(os.pathsep):
        path = path.strip()
        if not path:
            continue
        if os.path.isdir(path):
            site.addsitedir(path)


_append_extra_site_packages()
