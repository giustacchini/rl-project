"""Terrain, traffic rules and street bookkeeping (snowplow/road_map.py)."""

from pathlib import Path

import pytest

from Enums import ACTIONS, MovementActions
from snowplow.road_map import HIGHWAY, INTERSECTION, OBSTACLE, ROAD, RoadMap
from snowplow.road_network_configs import MAPS, road_network_1

LETTER = {MovementActions.UP: "U", MovementActions.DOWN: "D", MovementActions.LEFT: "L", MovementActions.RIGHT: "R"}


def moves_as_text(road_map: RoadMap) -> str:
    rows = []
    for y in range(road_map.height):
        row = []
        for x in range(road_map.width):
            if road_map.codes[y, x] == OBSTACLE:
                row.append("#")
            else:
                letters = sorted(LETTER[a] for i, a in enumerate(ACTIONS) if a in LETTER and road_map.allowed[y, x, i])
                row.append("".join(letters) or "-")
        rows.append(" ".join(f"{cell:>4}" for cell in row))
    return "\n".join(rows)


def test_traffic_rules_match_golden_table():
    """The rules of map 1 equal the table captured from the original implementation.

    (The only deliberate difference: the old code allowed UP from the intersection at x=11,y=4 into a
    wall; that phantom move was removed.)
    """
    golden = (Path(__file__).parents[2] / "golden" / "road_network_1_moves.txt").read_text()
    assert moves_as_text(RoadMap.from_config(road_network_1)) == golden


@pytest.mark.parametrize("name", sorted(MAPS))
def test_no_legal_move_enters_a_wall_or_leaves_the_map(name):
    road_map = RoadMap.from_config(MAPS[name])
    for y in range(road_map.height):
        for x in range(road_map.width):
            for i, action in enumerate(ACTIONS):
                if not road_map.allowed[y, x, i]:
                    continue
                dx, dy = action.value
                tx, ty = x + dx, y + dy
                assert 0 <= tx < road_map.width and 0 <= ty < road_map.height
                assert road_map.codes[ty, tx] != OBSTACLE


@pytest.mark.parametrize("name", sorted(MAPS))
def test_every_drivable_cell_can_wait_and_obstacles_allow_nothing(name):
    road_map = RoadMap.from_config(MAPS[name])
    wait = ACTIONS.index(MovementActions.DO_NOTHING)
    assert (road_map.allowed[..., wait] == road_map.passable).all()
    assert not road_map.allowed[~road_map.passable].any()


def test_highway_lanes_are_one_way_and_opposite():
    road_map = RoadMap.from_config(road_network_1)
    left, right = ACTIONS.index(MovementActions.LEFT), ACTIONS.index(MovementActions.RIGHT)
    # Rows 4 and 5 are the two lanes; x=5 is a plain highway cell (not an intersection).
    assert road_map.codes[4, 5] == HIGHWAY and road_map.codes[5, 5] == HIGHWAY
    assert road_map.allowed[4, 5, left] and not road_map.allowed[4, 5, right]  # northern lane flows west
    assert road_map.allowed[5, 5, right] and not road_map.allowed[5, 5, left]  # southern lane flows east
    for lane in (4, 5):  # exactly one real move on every highway lane cell
        assert road_map.allowed[lane, 5, 1:].sum() == 1


def test_intersection_allows_turning_and_switching_lane():
    road_map = RoadMap.from_config(road_network_1)
    up, down = ACTIONS.index(MovementActions.UP), ACTIONS.index(MovementActions.DOWN)
    assert road_map.codes[4, 3] == INTERSECTION
    assert road_map.allowed[4, 3, up] and road_map.allowed[4, 3, down]  # onto the street / to the other lane


@pytest.mark.parametrize("name", sorted(MAPS))
def test_streets_partition_the_drivable_cells(name):
    road_map = RoadMap.from_config(MAPS[name])
    assert ((road_map.street_id >= 0) == road_map.passable).all()
    assert sum(len(ys) for ys, _ in road_map.street_cells) == road_map.passable.sum()
    # main streets are made of highway / intersection cells only
    for sid, (ys, xs) in enumerate(road_map.street_cells):
        kinds = set(road_map.codes[ys, xs].tolist())
        assert kinds <= ({HIGHWAY, INTERSECTION} if road_map.street_is_main[sid] else {ROAD})


def test_invalid_maps_are_rejected():
    base = {"size": (3, 3), "roads": [{"position": (0, 0), "size": (3, 3)}], "highways": [], "intersections": [], "obstacles": []}
    RoadMap.from_config(base)  # valid

    with pytest.raises(ValueError, match="at least 3x3"):
        RoadMap.from_config(dict(base, size=(2, 2), roads=[{"position": (0, 0), "size": (2, 2)}]))

    overlap = dict(base, obstacles=[{"position": (0, 0), "size": (1, 1)}])
    with pytest.raises(ValueError, match="overlaps"):
        RoadMap.from_config(overlap)

    missing = dict(base, roads=[{"position": (0, 0), "size": (2, 3)}])
    with pytest.raises(ValueError, match="belong to no category"):
        RoadMap.from_config(missing)

    outside = dict(base, roads=[{"position": (0, 0), "size": (4, 4)}])
    with pytest.raises(ValueError, match="does not fit"):
        RoadMap.from_config(outside)

    # Two road columns separated by a wall: roads that a plow could never reach from the other side.
    split = {
        "size": (3, 3),
        "roads": [{"position": (0, 0), "size": (3, 1)}, {"position": (0, 2), "size": (3, 1)}],
        "highways": [],
        "intersections": [],
        "obstacles": [{"position": (0, 1), "size": (3, 1)}],
    }
    with pytest.raises(ValueError, match="not strongly connected"):
        RoadMap.from_config(split)
