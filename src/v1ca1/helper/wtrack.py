from __future__ import annotations

"""Shared animal-specific W-track geometry and linearization helpers.

This module centralizes hard-coded W-track geometry for animals whose implant
or track layout requires different node coordinates. The public API is built
around two lower-level concepts:

- branch side: `left` or `right`
- direction: `from_center` or `to_center`

Trajectory names such as `center_to_left` and `right_to_center` can be mapped
onto those two concepts with dedicated helpers. For example,
`center_to_left -> (branch_side="left", direction="from_center")` and
`left_to_center -> (branch_side="left", direction="to_center")`.

This lets downstream code represent both:

- natural trajectory-direction linearization
- branch-only linearization with a caller-chosen direction
"""

from typing import Any

import numpy as np


WTRACK_TRAJECTORY_TYPES = (
    "center_to_left",
    "left_to_center",
    "center_to_right",
    "right_to_center",
)
WTRACK_BRANCH_SIDES = ("left", "right")
WTRACK_DIRECTIONS = ("from_center", "to_center")

_WTRACK_BRANCH_SIDE_BY_TRAJECTORY = {
    "center_to_left": "left",
    "left_to_center": "left",
    "center_to_right": "right",
    "right_to_center": "right",
}
_WTRACK_DIRECTION_BY_TRAJECTORY = {
    "center_to_left": "from_center",
    "left_to_center": "to_center",
    "center_to_right": "from_center",
    "right_to_center": "to_center",
}

_WTRACK_GEOMETRY_BY_ANIMAL = {
    "L12": {
        "dx": 7.0,
        "dy": 9.0,
        "long_segment_length": 65.0,
        "short_segment_length": 21.5,
        "node_positions_right": np.array(
            [
                (59.5, 86.5),
                (59.5, 21.5),
                (52.5, 12.5),
                (31.0, 12.5),
                (24.0, 21.5),
                (24.0, 86.5),
            ]
        ),
        "node_positions_left": np.array(
            [
                (59.5, 86.5),
                (59.5, 21.5),
                (66.5, 12.5),
                (88.0, 12.5),
                (95.0, 21.5),
                (95.0, 86.5),
            ]
        ),
        "edges_from_center": np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        "edges_to_center": np.array([(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]),
        "edge_order_from_center": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        "edge_order_to_center": [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)],
    },
    "L14": {
        "dx": 9.5,
        "dy": 9.0,
        "long_segment_length": 81 - 17 - 2,
        "short_segment_length": 13.5,
        "node_positions_right": np.array(
            [
                (55.5, 81),
                (55.5, 81 - (81 - 17 - 2)),
                (55.5 - 9.5, 81 - (81 - 17 - 2) - 9.0),
                (55.5 - 9.5 - 13.5, 81 - (81 - 17 - 2) - 9.0),
                (55.5 - 2 * 9.5 - 13.5, 81 - (81 - 17 - 2)),
                (55.5 - 2 * 9.5 - 13.5, 81),
            ]
        ),
        "node_positions_left": np.array(
            [
                (55.5, 81),
                (55.5, 81 - (81 - 17 - 2)),
                (55.5 + 9.5, 81 - (81 - 17 - 2) - 9.0),
                (55.5 + 9.5 + 13.5, 81 - (81 - 17 - 2) - 9.0),
                (55.5 + 2 * 9.5 + 13.5, 81 - (81 - 17 - 2)),
                (55.5 + 2 * 9.5 + 13.5, 81),
            ]
        ),
        "edges_from_center": np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        "edges_to_center": np.array([(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]),
        "edge_order_from_center": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        "edge_order_to_center": [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)],
    },
    "L15": {
        "dx": 9.5,
        "dy": 9.0,
        "long_segment_length": 81 - 17 - 2,
        "short_segment_length": 13.5,
        "node_positions_right": np.array(
            [
                (55.5, 81),
                (55.5, 81 - (81 - 17 - 2)),
                (55.5 - 9.5, 81 - (81 - 17 - 2) - 9.0),
                (55.5 - 9.5 - 13.5, 81 - (81 - 17 - 2) - 9.0),
                (55.5 - 2 * 9.5 - 13.5, 81 - (81 - 17 - 2)),
                (55.5 - 2 * 9.5 - 13.5, 81),
            ]
        ),
        "node_positions_left": np.array(
            [
                (55.5, 81),
                (55.5, 81 - (81 - 17 - 2)),
                (55.5 + 9.5, 81 - (81 - 17 - 2) - 9.0),
                (55.5 + 9.5 + 13.5, 81 - (81 - 17 - 2) - 9.0),
                (55.5 + 2 * 9.5 + 13.5, 81 - (81 - 17 - 2)),
                (55.5 + 2 * 9.5 + 13.5, 81),
            ]
        ),
        "edges_from_center": np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        "edges_to_center": np.array([(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]),
        "edge_order_from_center": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        "edge_order_to_center": [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)],
    },
    "L16": {
        "dx": 9.5,
        "dy": 9.0,
        "long_segment_length": 81 - 17 - 2,
        "short_segment_length": 13.5,
        "node_positions_right": np.array(
            [
                (55.5, 81),
                (55.5, 81 - (81 - 17 - 2)),
                (55.5 - 9.5, 81 - (81 - 17 - 2) - 9.0),
                (55.5 - 9.5 - 13.5, 81 - (81 - 17 - 2) - 9.0),
                (55.5 - 2 * 9.5 - 13.5, 81 - (81 - 17 - 2)),
                (55.5 - 2 * 9.5 - 13.5, 81),
            ]
        ),
        "node_positions_left": np.array(
            [
                (55.5, 81),
                (55.5, 81 - (81 - 17 - 2)),
                (55.5 + 9.5, 81 - (81 - 17 - 2) - 9.0),
                (55.5 + 9.5 + 13.5, 81 - (81 - 17 - 2) - 9.0),
                (55.5 + 2 * 9.5 + 13.5, 81 - (81 - 17 - 2)),
                (55.5 + 2 * 9.5 + 13.5, 81),
            ]
        ),
        "edges_from_center": np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        "edges_to_center": np.array([(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]),
        "edge_order_from_center": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        "edge_order_to_center": [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)],
    },
    "L19": {
        "dx": 7.0,
        "dy": 7.0,
        "long_segment_length": 62.5,
        "short_segment_length": 17.0,
        "node_positions_right": np.array(
            [
                (52.5, 73.0),
                (52.5, 73.0 - 62.5),
                (52.5 - 7.0, 73.0 - 62.5 - 7.0),
                (52.5 - 7.0 - 17.0, 73.0 - 62.5 - 7.0),
                (52.5 - 2 * 7.0 - 17.0, 73.0 - 62.5),
                (52.5 - 2 * 7.0 - 17.0, 73.0),
            ]
        ),
        "node_positions_left": np.array(
            [
                (52.5, 73.0),
                (52.5, 73.0 - 62.5),
                (52.5 + 7.0, 73.0 - 62.5 - 7.0),
                (52.5 + 7.0 + 17.0, 73.0 - 62.5 - 7.0),
                (52.5 + 2 * 7.0 + 17.0, 73.0 - 62.5),
                (52.5 + 2 * 7.0 + 17.0, 73.0),
            ]
        ),
        "edges_from_center": np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        "edges_to_center": np.array([(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]),
        "edge_order_from_center": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        "edge_order_to_center": [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)],
    },
}


def get_wtrack_geometry(animal_name: str) -> dict[str, Any]:
    """Return the stored W-track geometry parameters for one animal."""
    if animal_name not in _WTRACK_GEOMETRY_BY_ANIMAL:
        available_animals = sorted(_WTRACK_GEOMETRY_BY_ANIMAL)
        raise ValueError(
            f"No W-track geometry is registered for animal {animal_name!r}. "
            f"Available animals: {available_animals!r}"
        )
    return _WTRACK_GEOMETRY_BY_ANIMAL[animal_name]


def get_wtrack_total_length(animal_name: str) -> float:
    """Return the total path length for one W-track trajectory."""
    geometry = get_wtrack_geometry(animal_name)
    diagonal_segment_length = np.sqrt(geometry["dx"] ** 2 + geometry["dy"] ** 2)
    return float(
        geometry["long_segment_length"] * 2
        + geometry["short_segment_length"]
        + 2 * diagonal_segment_length
    )


def get_wtrack_node_positions(
    animal_name: str,
    side: str | None = None,
) -> dict[str, np.ndarray] | np.ndarray:
    """Return stored W-track node positions for one animal."""
    geometry = get_wtrack_geometry(animal_name)
    node_positions = {
        "left": np.asarray(geometry["node_positions_left"], dtype=float),
        "right": np.asarray(geometry["node_positions_right"], dtype=float),
    }
    if side is None:
        return node_positions
    if side not in node_positions:
        raise ValueError("side must be one of {'left', 'right'} or None.")
    return node_positions[side]


def get_wtrack_branch_side(trajectory_type: str) -> str:
    """Return the branch side implied by one trajectory name."""
    if trajectory_type not in _WTRACK_BRANCH_SIDE_BY_TRAJECTORY:
        raise ValueError(
            f"Unknown trajectory_type {trajectory_type!r}. "
            f"Expected one of {WTRACK_TRAJECTORY_TYPES!r}."
        )
    return _WTRACK_BRANCH_SIDE_BY_TRAJECTORY[trajectory_type]


def get_wtrack_direction(trajectory_type: str) -> str:
    """Return the travel direction implied by one trajectory name."""
    if trajectory_type not in _WTRACK_DIRECTION_BY_TRAJECTORY:
        raise ValueError(
            f"Unknown trajectory_type {trajectory_type!r}. "
            f"Expected one of {WTRACK_TRAJECTORY_TYPES!r}."
        )
    return _WTRACK_DIRECTION_BY_TRAJECTORY[trajectory_type]


def get_wtrack_edges(
    animal_name: str,
    direction: str,
) -> np.ndarray:
    """Return the stored edge list for one travel direction."""
    geometry = get_wtrack_geometry(animal_name)
    if direction not in WTRACK_DIRECTIONS:
        raise ValueError(
            f"Unknown direction {direction!r}. Expected one of {WTRACK_DIRECTIONS!r}."
        )
    key = "edges_from_center" if direction == "from_center" else "edges_to_center"
    return np.asarray(geometry[key], dtype=int)


def get_wtrack_edge_order(
    animal_name: str,
    direction: str,
) -> list[tuple[int, int]]:
    """Return the linear edge order for one travel direction."""
    geometry = get_wtrack_geometry(animal_name)
    if direction not in WTRACK_DIRECTIONS:
        raise ValueError(
            f"Unknown direction {direction!r}. Expected one of {WTRACK_DIRECTIONS!r}."
        )
    key = (
        "edge_order_from_center"
        if direction == "from_center"
        else "edge_order_to_center"
    )
    return list(geometry[key])


def get_wtrack_branch_graph(
    animal_name: str,
    branch_side: str,
    direction: str,
) -> tuple[Any, list[tuple[int, int]]]:
    """Return a branch-specific W-track graph and edge order.

    This is the lower-level graph helper used by analyses that want to control
    branch choice and linearization direction independently.

    Parameters
    ----------
    animal_name : str
        Animal identifier with registered W-track geometry, for example `L14`.
    branch_side : str
        Which side branch to use: `left` or `right`.
    direction : str
        Which way to traverse that branch: `from_center` or `to_center`.
        For example, the natural `center_to_left` trajectory uses
        `branch_side="left"` with `direction="from_center"`, while
        `left_to_center` uses `branch_side="left"` with `direction="to_center"`.
    """
    import track_linearization as tl

    if branch_side not in WTRACK_BRANCH_SIDES:
        raise ValueError(
            f"Unknown branch_side {branch_side!r}. "
            f"Expected one of {WTRACK_BRANCH_SIDES!r}."
        )

    node_positions = np.asarray(get_wtrack_node_positions(animal_name, branch_side), dtype=float)
    edges = get_wtrack_edges(animal_name, direction)
    edge_order = get_wtrack_edge_order(animal_name, direction)
    return tl.make_track_graph(node_positions, edges), edge_order


def get_wtrack_full_graph(
    animal_name: str,
    branch_gap_cm: float = 15.0,
) -> tuple[Any, list[tuple[int, int]], list[float]]:
    """Return an animal-specific full W-track graph and linear edge spacing.

    The full graph is derived from the stored branch geometries instead of
    hard-coding a second coordinate table. Nodes 0 and 1 are the shared center
    well and center junction. Nodes 2-5 are the full left branch continuation,
    and nodes 6-9 are the full right branch continuation.
    """
    import track_linearization as tl

    if branch_gap_cm < 0:
        raise ValueError("--branch-gap-cm must be non-negative.")

    node_positions_by_side = get_wtrack_node_positions(animal_name)
    left_node_positions = np.asarray(node_positions_by_side["left"], dtype=float)
    right_node_positions = np.asarray(node_positions_by_side["right"], dtype=float)
    if left_node_positions.shape[0] < 6 or right_node_positions.shape[0] < 6:
        raise ValueError(
            "Full W-track construction expects six branch nodes per side. "
            f"Got {left_node_positions.shape[0]} left and "
            f"{right_node_positions.shape[0]} right nodes."
        )
    if not np.allclose(left_node_positions[:2], right_node_positions[:2]):
        raise ValueError(
            "Left and right W-track geometries do not share the same center "
            "well and center junction nodes."
        )

    node_positions = np.vstack(
        (
            left_node_positions[:2],
            left_node_positions[2:6],
            right_node_positions[2:6],
        )
    )
    edges = np.asarray(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (1, 6),
            (6, 7),
            (7, 8),
            (8, 9),
        ],
        dtype=int,
    )
    edge_order = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (1, 6),
        (6, 7),
        (7, 8),
        (8, 9),
    ]
    edge_spacing = [0.0, 0.0, 0.0, 0.0, float(branch_gap_cm), 0.0, 0.0, 0.0]
    return tl.make_track_graph(node_positions, edges), edge_order, edge_spacing
