# python
# shared/movement.py
"""
Generic simultaneous multi-agent movement resolution.

Build intentions, validate them, resolve conflicts, then commit positions.
Domain-specific restrictions are supplied through the movement hook.
"""
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
from mosaic_multigrid.base import AgentID

from Enums import MovementActions
from shared.grid import in_bounds, is_statically_blocked

# Type alias: (grid, current_pos, intended_pos) -> bool
MoveAllowedFn = Callable[[object, tuple, tuple], bool]


def build_movement_intentions(agents, actions: dict[AgentID, MovementActions]) -> dict:
    """
    Convert each agent's raw action into a movement record. No live
    agent state is modified here.

    Parameters
    ----------
    agents : list of Agent (mosaic_multigrid.core.agent.Agent)
    actions : {agent.index: raw_action}
    """
    movement_intentions = {}
    for agent in agents:
        raw_action: MovementActions = actions[agent.index]
        # agent.state.pos returns a plain tuple for a single agent's
        # AgentState slice (not an ndarray), so wrap it before .copy()/
        # arithmetic rather than relying on a numpy-only API.
        current_position = np.array(agent.state.pos, dtype=int)

        if raw_action.name in MovementActions.__members__:
            chosen_action = raw_action
            action_valid = True
        else:
            chosen_action = MovementActions.DO_NOTHING
            action_valid = False

        movement_intentions[agent.index] = {
            "raw_action": raw_action,
            "chosen_action": chosen_action,
            "action_valid": action_valid,
            "current_position": current_position,
            "intended_position": current_position + MovementActions(chosen_action).value,
        }
    return movement_intentions


def validate_proposed_moves(
    movement_intentions: dict, agents_by_id: dict, grid, width: int, height: int, is_move_allowed: Optional[MoveAllowedFn] = None
) -> dict:
    """
    Reject movements that are illegal before any conflict resolution:
    invalid raw actions, out-of-bounds targets, statically blocked
    targets (walls / forbidden cells), and anything the domain-specific
    `is_move_allowed` hook rejects (e.g. wrong-direction lane).
    """
    validated_moves = {agent_id: record.copy() for agent_id, record in movement_intentions.items()}

    for agent_id, record in validated_moves.items():
        record["status"] = "valid"
        current_pos = tuple(record["current_position"])
        intended_pos = tuple(record["intended_position"])

        blocked = not record["action_valid"] or not in_bounds(width, height, intended_pos) or is_statically_blocked(grid, intended_pos)

        if not blocked and current_pos != intended_pos and is_move_allowed is not None:
            if not is_move_allowed(grid, current_pos, intended_pos):
                blocked = True

        if blocked:
            record["status"] = "blocked_invalid_action"
            record["intended_position"] = record["current_position"]

    return validated_moves


def resolve_simultaneous_conflicts(validated_moves: dict, allow_shared_cells: bool = False) -> dict:
    """
    Resolve all agents from the same pre-step state (order-independent).

    With allow_shared_cells=False (exclusive cells, the default):
    - same destination requested by > 1 moving agent -> all blocked
    - two-agent swap -> both blocked
    - moving into a cell whose occupant stays -> blocked, propagated
      backwards through dependency chains
    - chains ending in an empty cell -> allowed

    With allow_shared_cells=True several agents may stand on one cell (e.g.
    snowplows working together on the same snowy cell). Agents then never
    block each other: every move that passed validation is executed, and the
    only remaining statuses are "moved" and "waited".
    """
    moves_after_conflicts = {agent_id: record.copy() for agent_id, record in validated_moves.items()}

    if allow_shared_cells:
        # No agent-agent conflicts exist when cells can be shared: just finalise the statuses.
        for record in moves_after_conflicts.values():
            record["resolved_position"] = record["intended_position"].copy()
            if record["status"] == "valid":
                same = tuple(record["current_position"]) == tuple(record["resolved_position"])
                record["status"] = "waited" if same else "moved"
        return moves_after_conflicts

    pos_list = defaultdict(list)
    pos_map = {}

    for agent_id, record in moves_after_conflicts.items():
        record["resolved_position"] = record["intended_position"].copy()
        pos_list[tuple(record["intended_position"])].append(agent_id)
        pos_map[tuple(record["current_position"])] = agent_id

    # 1. Same destination requested by more than one moving agent.
    for _, agent_ids in pos_list.items():
        moving_agent_ids = [
            agent_id
            for agent_id in agent_ids
            if tuple(moves_after_conflicts[agent_id]["intended_position"]) != tuple(moves_after_conflicts[agent_id]["current_position"])
        ]
        if len(moving_agent_ids) > 1:
            for agent_id in moving_agent_ids:
                record = moves_after_conflicts[agent_id]
                if record["status"] == "valid":
                    record["status"] = "blocked_same_destination"
                    record["resolved_position"] = record["current_position"].copy()

    # 2. Two-agent swaps.
    for agent_id, record in moves_after_conflicts.items():
        if record["status"] != "valid":
            continue
        current_pos = tuple(record["current_position"])
        intended_pos = tuple(record["intended_position"])
        if current_pos == intended_pos:
            continue

        occupant_id = pos_map.get(intended_pos)
        if occupant_id is None or occupant_id == agent_id:
            continue

        occupant = moves_after_conflicts[occupant_id]
        if occupant["status"] != "valid":
            continue

        if tuple(occupant["intended_position"]) == current_pos:
            record["status"] = "blocked_swap"
            record["resolved_position"] = record["current_position"].copy()
            occupant["status"] = "blocked_swap"
            occupant["resolved_position"] = occupant["current_position"].copy()

    # 3. Occupied-cell dependency propagation (repeat to a fixed point).
    changed = True
    while changed:
        changed = False
        for agent_id, record in moves_after_conflicts.items():
            if record["status"] != "valid":
                continue
            current_pos = tuple(record["current_position"])
            intended_pos = tuple(record["intended_position"])
            if current_pos == intended_pos:
                continue

            occupant_id = pos_map.get(intended_pos)
            if occupant_id is None or occupant_id == agent_id:
                continue

            occupant = moves_after_conflicts[occupant_id]
            occupant_current = tuple(occupant["current_position"])
            occupant_resolved = tuple(occupant["resolved_position"])

            if occupant_resolved == occupant_current:
                record["status"] = "blocked_occupied_cell"
                record["resolved_position"] = record["current_position"].copy()
                changed = True

    # 4. Final status for everything that survived.
    for _, record in moves_after_conflicts.items():
        if record["status"] != "valid":
            continue
        current_pos = tuple(record["current_position"])
        resolved_pos = tuple(record["resolved_position"])
        record["status"] = "waited" if current_pos == resolved_pos else "moved"

    return moves_after_conflicts


def commit_resolved_positions(agents_by_id: dict, moves_after_conflicts: dict) -> dict:
    """The only stage that writes resolved positions to live agent state."""
    for agent_id, record in moves_after_conflicts.items():
        agents_by_id[agent_id].state.pos = record["resolved_position"].copy()
    return moves_after_conflicts


def handle_simultaneous_movement(
    agents,
    grid,
    actions: dict[AgentID, MovementActions],
    width: int,
    height: int,
    is_move_allowed: Optional[MoveAllowedFn] = None,
    allow_shared_cells: bool = False,
) -> dict:
    """
    Convenience wrapper running all four stages in order, matching the
    `handle_actions()` lifecycle already used in SnowPlowEnv. Returns
    the final per-agent movement records (status, resolved_position, ...).

    allow_shared_cells: if True, agents may end the step on the same cell
    (see resolve_simultaneous_conflicts).
    """
    agents_by_id = {agent.index: agent for agent in agents}

    intentions = build_movement_intentions(agents, actions)
    validated = validate_proposed_moves(intentions, agents_by_id, grid, width, height, is_move_allowed)
    resolved = resolve_simultaneous_conflicts(validated, allow_shared_cells=allow_shared_cells)
    return commit_resolved_positions(agents_by_id, resolved)
