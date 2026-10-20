"""Grid2Op / L2RPN sandbox smoke environment (task m1-t13).

Used to validate the Grid2Op install end-to-end before building the custom
PettingZoo environment.

The factory prefers the LightSim2Grid backend (C++ Newton-Raphson, ~30x
faster than the default PandaPower backend) and falls back gracefully when
it is not installed. ``test=True`` uses the small chronics subset bundled
with grid2op (offline-friendly); ``test=False`` expects the full dataset in
``~/data_grid2op`` (downloaded on first call when network permits).
"""

from __future__ import annotations

DEFAULT_ENV = "l2rpn_case14_sandbox"


def make_sandbox(
    env_name: str = DEFAULT_ENV,
    *,
    test: bool = False,
    prefer_lightsim: bool = True,
    **kwargs,
):
    """Create a Grid2Op sandbox environment (m1-t13).

    Parameters
    ----------
    env_name:
        Grid2Op environment name. Defaults to the L2RPN IEEE-14 sandbox.
    test:
        If True, use the chronics subset shipped inside the grid2op package
        (no download needed). Keep False for real experiments.
    prefer_lightsim:
        Use the LightSim2Grid backend when available.
    kwargs:
        Forwarded verbatim to :func:`grid2op.make` (e.g. ``reward_class``).

    Returns
    -------
    grid2op.Environment.Environment
    """
    import grid2op

    backend = None
    if prefer_lightsim:
        try:
            from lightsim2grid import LightSimBackend

            backend = LightSimBackend()
        except ImportError:  # pragma: no cover - optional speed-up only
            backend = None

    if backend is not None:
        return grid2op.make(env_name, test=test, backend=backend, **kwargs)
    return grid2op.make(env_name, test=test, **kwargs)
