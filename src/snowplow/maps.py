"""Road layouts (maps) for the snowplow environment.

A map is a plain dict:

    "size":          (rows, columns)
    "roads":         secondary streets (two-way)
    "highways":      main-street lanes (one-way, see snowplow/road_map.py)
    "intersections": cells where a secondary street crosses the highway
    "obstacles":     buildings / blocked terrain

and every entry is a rectangle {"position": (row, column), "size": (rows, columns)}.
All four categories must tile the whole grid without overlapping.

NOTE ON "STREETS": each rectangle in "roads" counts as ONE secondary street and
each rectangle in "highways" / "intersections" counts as ONE main street. The
street-cleared rewards are given when all cells of such a rectangle are
snow-free, so splitting or merging rectangles changes what "a street" is.
"""

road_network_1 = {  # 10 x 19, 107 drivable cells: the main map of the project
    "size": (10, 19),
    "roads": [
        {"position": (0, 0), "size": (1, 1)},
        {"position": (0, 3), "size": (1, 1)},
        {"position": (0, 7), "size": (1, 1)},
        {"position": (0, 11), "size": (1, 1)},
        {"position": (0, 15), "size": (1, 1)},
        {"position": (0, 18), "size": (1, 1)},
        {"position": (1, 0), "size": (1, 4)},
        {"position": (1, 7), "size": (1, 12)},
        {"position": (2, 0), "size": (2, 1)},
        {"position": (2, 3), "size": (2, 1)},
        {"position": (2, 7), "size": (2, 1)},
        {"position": (2, 15), "size": (2, 1)},
        {"position": (2, 18), "size": (2, 1)},
        {"position": (6, 0), "size": (2, 1)},
        {"position": (6, 3), "size": (2, 1)},
        {"position": (6, 7), "size": (2, 1)},
        {"position": (6, 11), "size": (2, 1)},
        {"position": (6, 15), "size": (2, 1)},
        {"position": (6, 18), "size": (2, 1)},
        {"position": (8, 0), "size": (1, 19)},
        {"position": (9, 0), "size": (1, 1)},
        {"position": (9, 3), "size": (1, 1)},
        {"position": (9, 7), "size": (1, 1)},
        {"position": (9, 11), "size": (1, 1)},
        {"position": (9, 15), "size": (1, 1)},
        {"position": (9, 18), "size": (1, 1)},
    ],
    "highways": [
        {"position": (4, 1), "size": (2, 2)},
        {"position": (4, 4), "size": (2, 3)},
        {"position": (4, 8), "size": (2, 3)},
        {"position": (4, 12), "size": (2, 3)},
        {"position": (4, 16), "size": (2, 2)},
    ],
    "intersections": [
        {"position": (4, 0), "size": (2, 1)},
        {"position": (4, 3), "size": (2, 1)},
        {"position": (4, 7), "size": (2, 1)},
        {"position": (4, 11), "size": (2, 1)},
        {"position": (4, 15), "size": (2, 1)},
        {"position": (4, 18), "size": (2, 1)},
    ],
    "obstacles": [
        {"type": "wall", "position": (0, 1), "size": (1, 2)},
        {"type": "wall", "position": (0, 4), "size": (4, 3)},
        {"type": "wall", "position": (0, 8), "size": (1, 3)},
        {"type": "wall", "position": (0, 12), "size": (1, 3)},
        {"type": "wall", "position": (0, 16), "size": (1, 2)},
        {"type": "wall", "position": (2, 1), "size": (2, 2)},
        {"type": "wall", "position": (2, 8), "size": (2, 7)},
        {"type": "wall", "position": (2, 16), "size": (2, 2)},
        {"type": "wall", "position": (6, 1), "size": (2, 2)},
        {"type": "wall", "position": (6, 4), "size": (2, 3)},
        {"type": "wall", "position": (6, 8), "size": (2, 3)},
        {"type": "wall", "position": (6, 12), "size": (2, 3)},
        {"type": "wall", "position": (6, 16), "size": (2, 2)},
        {"type": "wall", "position": (9, 1), "size": (1, 2)},
        {"type": "wall", "position": (9, 4), "size": (1, 3)},
        {"type": "wall", "position": (9, 8), "size": (1, 3)},
        {"type": "wall", "position": (9, 12), "size": (1, 3)},
        {"type": "wall", "position": (9, 16), "size": (1, 2)},
    ],
}


# A tiny 8 x 9 map (42 drivable cells) with the same ingredients as map 1: one
# two-lane highway, secondary streets, intersections and dead ends. Use it to
# debug training in minutes before running on the big map.
road_network_small = {
    "size": (8, 9),
    "roads": [
        {"position": (0, 0), "size": (2, 1)},
        {"position": (0, 4), "size": (2, 1)},
        {"position": (0, 8), "size": (2, 1)},
        {"position": (4, 0), "size": (2, 1)},
        {"position": (4, 4), "size": (2, 1)},
        {"position": (4, 8), "size": (2, 1)},
        {"position": (6, 0), "size": (1, 9)},
        {"position": (7, 0), "size": (1, 1)},
        {"position": (7, 4), "size": (1, 1)},
        {"position": (7, 8), "size": (1, 1)},
    ],
    "highways": [
        {"position": (2, 1), "size": (2, 3)},
        {"position": (2, 5), "size": (2, 3)},
    ],
    "intersections": [
        {"position": (2, 0), "size": (2, 1)},
        {"position": (2, 4), "size": (2, 1)},
        {"position": (2, 8), "size": (2, 1)},
    ],
    "obstacles": [
        {"type": "wall", "position": (0, 1), "size": (2, 3)},
        {"type": "wall", "position": (0, 5), "size": (2, 3)},
        {"type": "wall", "position": (4, 1), "size": (2, 3)},
        {"type": "wall", "position": (4, 5), "size": (2, 3)},
        {"type": "wall", "position": (7, 1), "size": (1, 3)},
        {"type": "wall", "position": (7, 5), "size": (1, 3)},
    ],
}

# Name -> map, used by the command line (--map).
MAPS = {"network_1": road_network_1, "small": road_network_small}
