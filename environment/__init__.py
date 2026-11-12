from .kitchen import N_ACTIONS, PARTNERS, REWARDS, STAY, Kitchen

ENVS = ("kitchen", "warehouse", "foraging", "alem")


def make(env: str, *args, **kwargs):
    """env: 'kitchen' (Overcooked-AI), 'warehouse' (RWARE), 'foraging' (LBF) or 'alem' (Alem).
    Same interface."""
    if env == "kitchen":
        return Kitchen(*args, **kwargs)
    if env == "warehouse":
        from .warehouse import Warehouse
        return Warehouse(*args, **kwargs)
    if env == "foraging":
        from .foraging import Foraging
        return Foraging(*args, **kwargs)
    if env == "alem":
        from .alem import Alem
        return Alem(*args, **kwargs)
    raise ValueError(f"unknown env {env!r}; choose from {ENVS}")


__all__ = ["Kitchen", "ENVS", "make", "REWARDS", "PARTNERS", "N_ACTIONS", "STAY"]
