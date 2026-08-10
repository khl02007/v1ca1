"""Database-free execution of project Spyglass computations."""

from typing import Any


def run_figure_1_session(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazily run the initial offline Figure 1 analysis slice."""
    from v1ca1.spyglass.offline.figure_1 import (
        run_figure_1_session as _run_figure_1_session,
    )

    return _run_figure_1_session(*args, **kwargs)


def run_full_figure_1_session(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazily run one session's complete offline Figure 1 inputs."""
    from v1ca1.spyglass.offline.figure_1_full import (
        run_full_figure_session,
    )

    return run_full_figure_session(*args, **kwargs)


__all__ = ["run_figure_1_session", "run_full_figure_1_session"]
