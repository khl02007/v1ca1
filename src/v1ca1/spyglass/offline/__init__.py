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


def run_figure_2_session(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazily run one session's complete offline Figure 2 inputs."""
    from v1ca1.spyglass.offline.figure_2 import (
        run_figure_2_session as _run_figure_2_session,
    )

    return _run_figure_2_session(*args, **kwargs)


def run_figure_3_session(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazily run one session's complete offline Figure 3 inputs."""
    from v1ca1.spyglass.offline.figure_3 import (
        run_figure_3_session as _run_figure_3_session,
    )

    return _run_figure_3_session(*args, **kwargs)


def build_figure_3_schematic_supplement(
    *args: Any, **kwargs: Any
) -> dict[str, Any]:
    """Lazily build one immutable Figure 3 schematic supplement."""
    from v1ca1.spyglass.offline.figure_3_schematic_supplement import (
        build_figure_3_schematic_supplement as _build_supplement,
    )

    return _build_supplement(*args, **kwargs)


__all__ = [
    "build_figure_3_schematic_supplement",
    "run_figure_1_session",
    "run_figure_2_session",
    "run_figure_3_session",
    "run_full_figure_1_session",
]
