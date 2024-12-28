from omegaconf import OmegaConf

from .paths import find_project_root


def register_custom_resolvers():
    OmegaConf.register_new_resolver(
        "find_project_root", lambda: str(find_project_root()), use_cache=True
    )
