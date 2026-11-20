"""Two environments, same interface: overcooked (environment/overcooked.py) and rware (environment/rware.py).
The packages are imported only when an environment is made, so `from environment import canon` stays light."""

ENVS = ("overcooked", "rware")
ALIASES = {"kitchen": "overcooked", "warehouse": "rware"}   # old names, still found in older results folders


def canon(env: str) -> str:
    """Environment name with the old names mapped to the new ones: kitchen -> overcooked, warehouse -> rware."""
    return ALIASES.get(env, env)


def make(env: str, *args, **kwargs):
    """env: 'overcooked' (Overcooked-AI) or 'rware' (multi-robot warehouse)."""
    env = canon(env)
    if env == "overcooked":
        from .overcooked import Overcooked
        return Overcooked(*args, **kwargs)
    if env == "rware":
        from .rware import Rware
        return Rware(*args, **kwargs)
    raise ValueError(f"unknown env {env!r}; choose from {ENVS}")


def __getattr__(name):                      # environment.Overcooked, environment.REWARDS, ... on first use
    if name in ("Overcooked", "N_ACTIONS", "PARTNERS", "REWARDS", "STAY"):
        from . import overcooked
        return getattr(overcooked, name)
    if name == "Rware":
        from .rware import Rware
        return Rware
    raise AttributeError(name)


__all__ = ["ENVS", "ALIASES", "canon", "make", "Overcooked", "Rware", "REWARDS", "PARTNERS", "N_ACTIONS", "STAY"]
