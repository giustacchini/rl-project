# python
# snow_plow_env.py

import enum
from collections import defaultdict

from mosaic_multigrid.base import AgentID, MultiGridEnv


class SnowPlowActions(enum.Enum):
    DO_NOTHING = (0, 0)
    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)


class SnowPlowEnv(MultiGridEnv):

    def __init__(self, width: int, height: int, num_agents: int):
        super().__init__(width, height, num_agents)

    def handle_actions(self, actions):
        movement_intentions = self.build_movement_intentions(actions)
        validated_moves = self.validate_proposed_moves(movement_intentions)
        moves_after_conflicts = self.resolve_simultaneous_conflicts(validated_moves)
        movement_results = self.commit_resolved_positions(moves_after_conflicts)

        self.last_movement_results = movement_results

        rewards = {agent.index: 0 for agent in self.agents}
        return rewards

    def build_movement_intentions(self, actions: dict[AgentID, SnowPlowActions]):
        movement_intentions = {}
        for agent in self.agents:
            raw_action = actions[agent.index]
            movement_intentions[agent.index] = {"raw_action": raw_action, "current_position": agent.state.pos.copy()}

            # General approach to handle invalid actions, including non-integer values and out-of-range integers
            try:
                movement_intentions[agent.index]["chosen_action"] = SnowPlowActions(raw_action)
                movement_intentions[agent.index]["action_valid"] = True
            except (ValueError, TypeError):
                movement_intentions[agent.index]["chosen_action"] = SnowPlowActions.DO_NOTHING
                movement_intentions[agent.index]["action_valid"] = False

            chosen_action = movement_intentions[agent.index]["chosen_action"]
            movement_intentions[agent.index]["intended_position"] = (
                movement_intentions[agent.index]["current_position"] + SnowPlowActions(chosen_action).value
            )

        return movement_intentions

    def validate_proposed_moves(self, movement_intentions):
        validated_moves = {agent_id: record.copy() for agent_id, record in movement_intentions.items()}
        for agent in self.agents:
            validated_moves[agent.index]["status"] = "valid"
            if (
                validated_moves[agent.index]["action_valid"] is False
                or validated_moves[agent.index]["intended_position"][0] < 0  # Trying to move out of bounds (LEFT)
                or validated_moves[agent.index]["intended_position"][0] >= self.width  # Trying to move out of bounds (RIGHT)
                or validated_moves[agent.index]["intended_position"][1] < 0  # Trying to move out of bounds (UP)
                or validated_moves[agent.index]["intended_position"][1] >= self.height  # Trying to move out of bounds (DOWN)
            ):
                validated_moves[agent.index]["status"] = "blocked_invalid_action"
                validated_moves[agent.index]["intended_position"] = validated_moves[agent.index]["current_position"]
        return validated_moves

    def resolve_simultaneous_conflicts(self, validated_moves):
        moves_after_conflicts = {agent_id: record.copy() for agent_id, record in validated_moves.items()}

        pos_list = defaultdict(list)
        pos_map = {}

        # Initialize resolved positions and create position mappings
        for agent_id, record in moves_after_conflicts.items():
            record["resolved_position"] = record["intended_position"].copy()

            pos_list[tuple(record["intended_position"])].append(agent_id)
            pos_map[tuple(record["current_position"])] = agent_id

        # 1. Detect multiple moving agents requesting the same destination
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

        # 2. Detect two-agent swaps
        for agent_id, record in moves_after_conflicts.items():
            if record["status"] != "valid":
                continue

            current_pos = tuple(record["current_position"])
            intended_pos = tuple(record["intended_position"])

            # Agent is not trying to move
            if current_pos == intended_pos:
                continue

            occupant_id = pos_map.get(intended_pos)

            # Destination is empty
            if occupant_id is None or occupant_id == agent_id:
                continue

            occupant = moves_after_conflicts[occupant_id]

            # Only consider a swap if the other agent is still allowed to move
            if occupant["status"] != "valid":
                continue

            occupant_intended_pos = tuple(occupant["intended_position"])

            # A wants B's cell and B wants A's cell
            if occupant_intended_pos == current_pos:
                record["status"] = "blocked_swap"
                record["resolved_position"] = record["current_position"].copy()

                occupant["status"] = "blocked_swap"
                occupant["resolved_position"] = occupant["current_position"].copy()

        # 3. Resolve occupied-cell dependencies
        #
        # Repeat until no new agent becomes blocked.
        # Example:
        # A -> B -> C -> D
        # If D stays, C is blocked, then B is blocked, then A is blocked.
        changed = True

        while changed:
            changed = False

            for agent_id, record in moves_after_conflicts.items():
                if record["status"] != "valid":
                    continue

                current_pos = tuple(record["current_position"])
                intended_pos = tuple(record["intended_position"])

                # Agent intentionally stays
                if current_pos == intended_pos:
                    continue

                occupant_id = pos_map.get(intended_pos)

                # Empty destination
                if occupant_id is None or occupant_id == agent_id:
                    continue

                occupant = moves_after_conflicts[occupant_id]

                occupant_current = tuple(occupant["current_position"])
                occupant_resolved = tuple(occupant["resolved_position"])

                # The occupying agent is not leaving the cell
                if occupant_resolved == occupant_current:
                    record["status"] = "blocked_occupied_cell"
                    record["resolved_position"] = record["current_position"].copy()
                    changed = True

        # 4. Assign final statuses to movements that survived conflict resolution
        for _, record in moves_after_conflicts.items():
            if record["status"] != "valid":
                continue

            current_pos = tuple(record["current_position"])
            resolved_pos = tuple(record["resolved_position"])

            if current_pos == resolved_pos:
                record["status"] = "waited"
            else:
                record["status"] = "moved"

        return moves_after_conflicts

    def commit_resolved_positions(self, moves_after_conflicts):
        for agent_id, record in moves_after_conflicts.items():
            self.agents[agent_id].state.pos = record["resolved_position"].copy()
        return moves_after_conflicts
