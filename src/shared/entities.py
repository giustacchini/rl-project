# python
# shared/entities.py
"""
Generic "entity" building blocks.

Issue #3 asks for "generic Entity objects." mosaic_multigrid already has
exactly this: `WorldObj` (mosaic_multigrid.core.world_object), with
concrete subclasses Wall, Floor, Goal, Lava, Door, Key, Ball, Box -- each
already answering `can_overlap()` / `can_pickup()`. Writing a second,
parallel Entity class would duplicate that system and fight the Grid
class that already stores WorldObj instances.

Instead, this module just re-exports the pieces every domain actually
needs, under names that read naturally in snowplow/soccer code, so
neither domain has to import mosaic_multigrid internals directly.
"""

from mosaic_multigrid.core.world_object import WorldObj, Wall, Floor, Goal

# A generic blocking cell (forbidden cell / obstacle). `Wall.can_overlap()`
# is False, so anything placed here blocks both static validity checks
# (shared/grid.py) and the shared movement resolver.
Obstacle = Wall

# A generic walkable marker cell (e.g. "this is a priority road" or
# "this is the field"), left un-opinionated: can_overlap() is True so it
# never blocks movement on its own. Domains can subclass this if they
# need to tag cells with extra meaning later.
Walkable = Floor

__all__ = ["WorldObj", "Obstacle", "Walkable", "Wall", "Floor", "Goal"]
