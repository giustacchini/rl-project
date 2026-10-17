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

road_network_2 = {  # 20 x 19, 107 drivable cells: the main map of the project
    "size": (20, 19),
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
        {"position": (10, 0), "size": (1, 1)},
        {"position": (10, 3), "size": (1, 1)},
        {"position": (10, 7), "size": (1, 1)},
        {"position": (10, 11), "size": (1, 1)},
        {"position": (10, 15), "size": (1, 1)},
        {"position": (10, 18), "size": (1, 1)},
        {"position": (11, 0), "size": (1, 4)},
        {"position": (11, 7), "size": (1, 12)},
        {"position": (12, 0), "size": (2, 1)},
        {"position": (12, 3), "size": (2, 1)},
        {"position": (12, 7), "size": (2, 1)},
        {"position": (12, 15), "size": (2, 1)},
        {"position": (12, 18), "size": (2, 1)},
        {"position": (16, 0), "size": (2, 1)},
        {"position": (16, 3), "size": (2, 1)},
        {"position": (16, 7), "size": (2, 1)},
        {"position": (16, 11), "size": (2, 1)},
        {"position": (16, 15), "size": (2, 1)},
        {"position": (16, 18), "size": (2, 1)},
        {"position": (18, 0), "size": (1, 19)},
        {"position": (19, 0), "size": (1, 1)},
        {"position": (19, 3), "size": (1, 1)},
        {"position": (19, 7), "size": (1, 1)},
        {"position": (19, 11), "size": (1, 1)},
        {"position": (19, 15), "size": (1, 1)},
        {"position": (19, 18), "size": (1, 1)},
    ],
    "highways": [
        {"position": (4, 1), "size": (2, 2)},
        {"position": (4, 4), "size": (2, 3)},
        {"position": (4, 8), "size": (2, 3)},
        {"position": (4, 12), "size": (2, 3)},
        {"position": (4, 16), "size": (2, 2)},
        {"position": (14, 1), "size": (2, 2)},
        {"position": (14, 4), "size": (2, 3)},
        {"position": (14, 8), "size": (2, 3)},
        {"position": (14, 12), "size": (2, 3)},
        {"position": (14, 16), "size": (2, 2)},
    ],
    "intersections": [
        {"position": (4, 0), "size": (2, 1)},
        {"position": (4, 3), "size": (2, 1)},
        {"position": (4, 7), "size": (2, 1)},
        {"position": (4, 11), "size": (2, 1)},
        {"position": (4, 15), "size": (2, 1)},
        {"position": (4, 18), "size": (2, 1)},
        {"position": (14, 0), "size": (2, 1)},
        {"position": (14, 3), "size": (2, 1)},
        {"position": (14, 7), "size": (2, 1)},
        {"position": (14, 11), "size": (2, 1)},
        {"position": (14, 15), "size": (2, 1)},
        {"position": (14, 18), "size": (2, 1)},
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
        {"type": "wall", "position": (10, 1), "size": (1, 2)},
        {"type": "wall", "position": (10, 4), "size": (4, 3)},
        {"type": "wall", "position": (10, 8), "size": (1, 3)},
        {"type": "wall", "position": (10, 12), "size": (1, 3)},
        {"type": "wall", "position": (10, 16), "size": (1, 2)},
        {"type": "wall", "position": (12, 1), "size": (2, 2)},
        {"type": "wall", "position": (12, 8), "size": (2, 7)},
        {"type": "wall", "position": (12, 16), "size": (2, 2)},
        {"type": "wall", "position": (16, 1), "size": (2, 2)},
        {"type": "wall", "position": (16, 4), "size": (2, 3)},
        {"type": "wall", "position": (16, 8), "size": (2, 3)},
        {"type": "wall", "position": (16, 12), "size": (2, 3)},
        {"type": "wall", "position": (16, 16), "size": (2, 2)},
        {"type": "wall", "position": (19, 1), "size": (1, 2)},
        {"type": "wall", "position": (19, 4), "size": (1, 3)},
        {"type": "wall", "position": (19, 8), "size": (1, 3)},
        {"type": "wall", "position": (19, 12), "size": (1, 3)},
        {"type": "wall", "position": (19, 16), "size": (1, 2)},
    ],
}

road_network_3 = {
    # (ROW, COL)
    "size": (20, 19),
    "roads": [
        # Top row singles
        {"position": (0, 0), "size": (1, 1)},
        {"position": (0, 3), "size": (1, 1)},
        {"position": (0, 7), "size": (1, 1)},
        {"position": (0, 15), "size": (1, 1)},
        {"position": (0, 18), "size": (1, 1)},
        # Upper horizontal road
        {"position": (1, 0), "size": (1, 4)},
        {"position": (1, 7), "size": (1, 4)},
        {"position": (1, 13), "size": (1, 6)},
        # Vertical roads
        {"position": (2, 0), "size": (2, 1)},
        {"position": (2, 3), "size": (2, 1)},
        {"position": (2, 7), "size": (2, 1)},
        {"position": (2, 15), "size": (2, 1)},
        {"position": (2, 18), "size": (2, 1)},
        # Middle vertical roads
        {"position": (6, 0), "size": (2, 1)},
        {"position": (6, 3), "size": (2, 1)},
        {"position": (6, 7), "size": (2, 1)},
        {"position": (6, 15), "size": (2, 1)},
        {"position": (6, 18), "size": (2, 1)},
        # Middle horizontal roads
        {"position": (8, 0), "size": (1, 11)},
        {"position": (8, 13), "size": (1, 6)},
        {"position": (11, 0), "size": (1, 19)},
        # Lower middle vertical roads
        {"position": (9, 0), "size": (2, 1)},
        {"position": (9, 3), "size": (2, 1)},
        {"position": (9, 7), "size": (2, 1)},
        {"position": (9, 15), "size": (2, 1)},
        {"position": (9, 18), "size": (2, 1)},
        # 2nd lower middle vertical roads
        {"position": (12, 0), "size": (2, 1)},
        {"position": (12, 3), "size": (2, 1)},
        {"position": (12, 7), "size": (2, 1)},
        # Upper bottom vertical roads
        {"position": (16, 0), "size": (2, 1)},
        {"position": (16, 3), "size": (2, 1)},
        {"position": (16, 7), "size": (2, 1)},
        {"position": (16, 11), "size": (2, 1)},
        # Medium vertical roads
        {"position": (12, 15), "size": (6, 1)},
        {"position": (12, 18), "size": (6, 1)},
        # Highway connections
        {"position": (15, 8), "size": (1, 3)},
        {"position": (15, 13), "size": (1, 2)},
        # Lower horizontal road
        {"position": (18, 0), "size": (1, 19)},
        # Bottom row singles
        {"position": (19, 0), "size": (1, 1)},
        {"position": (19, 3), "size": (1, 1)},
        {"position": (19, 7), "size": (1, 1)},
        {"position": (19, 11), "size": (1, 1)},
        {"position": (19, 15), "size": (1, 1)},
        {"position": (19, 18), "size": (1, 1)},
    ],
    "highways": [
        # Horizontal highway across the upper-middle section
        {"position": (2, 11), "size": (2, 2)},
        {"position": (6, 11), "size": (2, 2)},
        {"position": (9, 11), "size": (2, 2)},
        {"position": (12, 11), "size": (3, 2)},
        # Vertical highway
        {"position": (4, 1), "size": (2, 2)},
        {"position": (4, 4), "size": (2, 3)},
        {"position": (4, 8), "size": (2, 3)},
        {"position": (4, 13), "size": (2, 2)},
        {"position": (4, 16), "size": (2, 2)},
        # Lower horizontal highway, only extending across the central section
        {"position": (14, 1), "size": (2, 2)},
        {"position": (14, 4), "size": (2, 3)},
    ],
    "intersections": [
        # Upper horizontal highway intersections
        {"position": (4, 0), "size": (2, 1)},
        {"position": (4, 3), "size": (2, 1)},
        {"position": (4, 7), "size": (2, 1)},
        {"position": (4, 11), "size": (2, 2)},
        {"position": (4, 15), "size": (2, 1)},
        {"position": (4, 18), "size": (2, 1)},
        # Vertical highway intersections
        {"position": (1, 11), "size": (1, 2)},
        {"position": (8, 11), "size": (1, 2)},
        {"position": (11, 11), "size": (1, 2)},
        {"position": (15, 11), "size": (1, 2)},
        # Lower horizontal highway intersections
        {"position": (14, 0), "size": (2, 1)},
        {"position": (14, 3), "size": (2, 1)},
        {"position": (14, 7), "size": (2, 1)},
        {"position": (14, 11), "size": (2, 1)},
    ],
}


# Name -> map, used by the command line (--map).
MAPS = {
    "network_1": road_network_1,
    "network_2": road_network_2,
    "network_3": road_network_3,
}
