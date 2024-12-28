import functools
from pathlib import Path

import rootutils


def get_config_path() -> Path:
    return find_project_root() / "configs"


@functools.cache
def find_project_root() -> Path:
    return rootutils.find_root(search_from=__file__, indicator=".project-root")
