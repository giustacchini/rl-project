"""The snowplow environment: N plows clear snow from a road network, all acting at once.

HOW ONE EPISODE RUNS (this is the main loop of the whole project)

    obs, info = env.reset(seed=0)
    while not done:
        actions = {plow_id: policy(obs[plow_id]) for plow_id in obs}     # decide
        obs, rewards, terminated, truncated, info = env.step(actions)    # act

`env.step()` runs the lifecycle below. All plows move SIMULTANEOUSLY: nobody is
processed before anybody else, conflicts are resolved from the same starting state.

    1. receive the actions            (one per plow, a dict {plow_id: action})
    2. validate the actions           (malformed actions raise; a missing plow waits)
    3. compute intended movements     \\
    4. detect conflicts                } shared/movement.py (same code the soccer env will use)
    5. resolve movements              /
    6. resolve interactions           plows standing on snowy cells plow them (time-based, cooperative)
    7. update dynamic world state     clouds get older, expired clouds vanish
    8. trigger stochastic events      a new cloud may appear; every cloud snows on the roads below
    9. calculate rewards              from the events of this step
   10. check termination/truncation   success = every road clean; truncation = max_steps
   11. generate observations
   12. return (obs, rewards, terminated, truncated, info), each a dict {plow_id: value}

COOPERATIVE PLOWING. Plows may share a cell (config.allow_shared_cells). A plow that stands on
a snowy cell is BUSY: it stays there (its only legal action is DO_NOTHING) until the cell is
completely clean. Cleaning a cell takes config.snow.plow_time(level, n_plows) steps, e.g. with
the defaults one plow needs 1 / 4 / 8 steps for snow level 1 / 2 / 3 and two plows on the same
cell need only 1 / 2 / 3 steps. The step on which a plow arrives already counts as work.

ACTIONS. A policy outputs an integer i and the env executes Enums.ACTIONS[i]
(DO_NOTHING, UP, DOWN, RIGHT, LEFT). Not every action is legal on every cell (walls,
one-way highway lanes). `obs[plow]["action_mask"][i]` is True when action i is legal,
so a learner can restrict itself to legal moves instead of wasting steps on illegal ones.
Illegal actions are still accepted: the plow stays where it is and gets a penalty.

STATE LIVES IN NUMPY, NOT IN CELL OBJECTS
    road_map   (static)  terrain, traffic rules, streets         -> snowplow/road_map.py
    snow_field (dynamic) snow levels, clouds                      -> snowplow/snow.py
mosaic's grid and `GridCell` objects only draw the picture and block walls.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from mosaic_multigrid.base import AgentID, MultiGridEnv
from mosaic_multigrid.core.grid import Grid
from mosaic_multigrid.core.world_object import WorldObj

from Enums import ACTIONS, EventType, MovementActions, SnowplowEvent
from shared.events import EventLog
from shared.metrics import EpisodeMetrics
from shared.movement import handle_simultaneous_movement
from shared.observations import FullObservationProvider, LocalObservationProvider
from snowplow import road_map as rm
from snowplow.cell import GridCell
from snowplow.config import SnowplowConfig
from snowplow.encoding import encode_state
from snowplow.rewards import make_snowplow_reward_calculator
from snowplow.road_map import RoadMap
from snowplow.road_network_configs import road_network_1
from snowplow.snow import SnowField

# Channels of the observable grid, in order (see get_observable_grid).
# "plow_progress" (last channel) is the fraction of plowing work already done on every cell, so a
# plow can see whether a teammate is busy on a cell and how much work is left there.
GRID_CHANNELS = (
    "snow", "obstacle", "road", "highway", "intersection", "can_up", "can_down", "can_right", "can_left", "plow_progress",
)

# mosaic's Direction numbering (used only so the drawn plow faces where it drove).
_FACING = {MovementActions.RIGHT: 0, MovementActions.DOWN: 1, MovementActions.LEFT: 2, MovementActions.UP: 3}

# Movement statuses (from shared/movement.py) that count as a collision between plows.
_COLLISION_STATUSES = {"blocked_same_destination", "blocked_swap", "blocked_occupied_cell"}


class SnowPlowEnv(MultiGridEnv):
    """Multi-agent snowplow environment (PettingZoo-parallel style API, agent ids are 0..N-1)."""

    def __init__(
        self,
        road_network: dict | None = None,
        config: SnowplowConfig | None = None,
        render_mode: str | None = None,
        screen_size: int = 500,
        render_fps: int = 20,
        **kwargs,
    ):
        self.metadata["render_fps"] = render_fps
        self.config = config or SnowplowConfig()
        self.road_map = RoadMap.from_config(road_network or road_network_1)
        self.snow_field = SnowField(self.road_map, self.config.snow)
        self.event_log = EventLog()
        self.metrics = EpisodeMetrics()
        self.reward_calculator = make_snowplow_reward_calculator()
        self.observation_provider = (
            FullObservationProvider() if self.config.observation == "full" else LocalObservationProvider(self.config.view_size)
        )
        self.last_movement_results: dict = {}
        self._finished = False
        # Cooperative plowing bookkeeping: for every cell that is being plowed, how much of its
        # work each plow has done so far ({(x, y): {plow_id: work fraction}}). Used to split the
        # clearing reward fairly when several plows worked on the same cell.
        self._work_share: dict[tuple[int, int], dict[AgentID, float]] = {}

        # Static channels never change during an episode: compute them once.
        codes = self.road_map.codes
        self._static_channels = np.stack(
            [codes == code for code in (rm.OBSTACLE, rm.ROAD, rm.HIGHWAY, rm.INTERSECTION)]
            + [
                self.road_map.allowed[..., ACTIONS.index(move)]
                for move in (MovementActions.UP, MovementActions.DOWN, MovementActions.RIGHT, MovementActions.LEFT)
            ]
        ).astype(
            np.float32
        )  # [8, H, W]

        super().__init__(
            mission_space="clear the snow from every road",
            agents=self.config.n_plows,
            width=self.road_map.width,
            height=self.road_map.height,
            max_steps=self.config.max_steps,
            render_mode=render_mode,
            screen_size=screen_size,
            highlight=False,  # mosaic's field-of-view highlight does not apply to this env
            **kwargs,
        )

    # ------------------------------------------------------------ properties
    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    @property
    def possible_agents(self) -> list[AgentID]:
        """Agent ids in a fixed order (PettingZoo naming)."""
        return [agent.index for agent in self.agents]

    def agent_positions(self) -> np.ndarray:
        """[N, 2] array of (x, y) of every plow, in id order (used by reward shaping)."""
        return np.array([tuple(agent.state.pos) for agent in self.agents], dtype=np.int64)

    # ----------------------------------------------------------------- reset
    def reset(self, seed: int | None = None, **kwargs):
        """Start a new episode. Returns (observations, infos), each a dict {plow_id: ...}.

        mosaic's reset seeds `self.np_random`, calls `_gen_grid` (below) to build the world
        and then `gen_obs` to produce the first observations.
        """
        observations, _ = super().reset(seed=seed, **kwargs)
        return observations, {agent.index: {} for agent in self.agents}

    def _gen_grid(self, width: int, height: int):
        """Build the world of a new episode: draw the map, place plows, let the first snow fall."""
        self.grid = Grid(width, height)
        for y in range(height):
            for x in range(width):
                self.grid.set(x, y, GridCell(int(self.road_map.codes[y, x])))

        for agent in self.agents:
            self.place_agent(agent)  # random free drivable cell (see place_obj)

        self.snow_field.reset(self.np_random)
        self.event_log.clear()
        self.metrics.reset()
        self.last_movement_results = {}
        self._finished = False
        self._work_share = {}

    def place_obj(
        self,
        obj: WorldObj | None,
        top: tuple[int, int] | None = None,
        size: tuple[int, int] | None = None,
        reject_fn: Callable[[MultiGridEnv, tuple[int, int]], bool] | None = None,
        max_tries=math.inf,
    ) -> tuple[int, int]:
        """Pick a random drivable cell without another plow on it (used to place plows).

        Replaces mosaic's rejection sampling, which does not know about our terrain. Objects
        (obj is not None) are placed by mosaic as usual; `top`, `size` and `max_tries` only
        matter for that case.
        """
        if obj is not None:
            return super().place_obj(obj, top, size, reject_fn, max_tries)

        free = [
            (int(x), int(y))
            for y, x in np.argwhere(self.road_map.passable)
            if not np.all(self.agent_states.pos == (x, y), axis=1).any() and not (reject_fn and reject_fn(self, (int(x), int(y))))
        ]
        if not free:
            raise RuntimeError("no free drivable cell left to place a plow")
        return free[int(self.np_random.integers(len(free)))]

    # ---------------------------------------------------------- what plows see
    def get_observable_grid(self) -> np.ndarray:
        """Everything the plows may see, as a float array [10, H, W] (see GRID_CHANNELS).

        snow/max_snow, terrain one-hot (obstacle, road, highway, intersection), which of
        the four moves is allowed from each cell (= lane directions) and the plowing progress
        (0..1) of every cell. Clouds are NOT in here: they are hidden from the plows.
        """
        grid = np.empty((len(GRID_CHANNELS), self.height, self.width), dtype=np.float32)
        grid[0] = self.snow_field.snow / self.config.snow.max_snow
        grid[1:9] = self._static_channels
        grid[9] = self.snow_field.plow_progress
        return grid

    def is_busy(self, agent_id: AgentID) -> bool:
        """True while the plow stands on a snowy cell: it is plowing and may not leave yet."""
        x, y = (int(v) for v in self.agents[agent_id].state.pos)
        return bool(self.snow_field.snow[y, x] > 0)

    def get_action_mask(self, agent_id: AgentID) -> np.ndarray:
        """bool [n_actions]: which actions are legal on the plow's current cell (DO_NOTHING always is).

        A busy plow (on a snowy cell) can only DO_NOTHING: it keeps plowing until the cell is clean.
        """
        if self.is_busy(agent_id):
            mask = np.zeros(len(ACTIONS), dtype=bool)
            mask[ACTIONS.index(MovementActions.DO_NOTHING)] = True
            return mask
        x, y = (int(v) for v in self.agents[agent_id].state.pos)
        return self.road_map.allowed[y, x].copy()

    def gen_obs(self):
        """Observations of all plows. (Overrides mosaic's egocentric image observation.)"""
        return self.observation_provider.get_observations(self)

    def state(self) -> np.ndarray:
        """Global state vector for centralised critics (includes everything plows can see)."""
        return encode_state(self)

    # ------------------------------------------------------------------ step
    def step(self, actions: dict[AgentID, MovementActions | int]):
        """Advance the world by one step. See the module docstring for the 12 stages."""
        if self._finished:
            raise RuntimeError("the episode is over: call reset() before step()")
        self.step_count += 1
        self.event_log.clear()

        # 1-2. receive + validate
        action_of = self._validate_actions(actions)

        # 2b. busy plows (standing on a snowy cell) stay where they are and keep plowing,
        #     whatever action they sent (their action mask only allows DO_NOTHING anyway).
        busy = {agent.index for agent in self.agents if self.is_busy(agent.index)}
        for agent_id in busy:
            action_of[agent_id] = MovementActions.DO_NOTHING

        # 3-5. intended movements -> conflicts -> resolution (shared simultaneous-movement code)
        results = handle_simultaneous_movement(
            agents=self.agents,
            grid=self.grid,
            actions=action_of,
            width=self.width,
            height=self.height,
            is_move_allowed=self.is_move_allowed,
            allow_shared_cells=self.config.allow_shared_cells,
        )
        for agent_id in busy:
            results[agent_id]["status"] = "plowing"
        self.last_movement_results = results
        self._emit_movement_events(results)

        # 6. interactions: every plow standing on a snowy cell works on it (alone or together)
        self._plow_snow(results)

        # 7. update dynamic world state, 8. stochastic events
        self.snow_field.age_clouds()
        self.snow_field.snowfall_step(self.np_random, self.step_count)

        # 10a. success: the roads are clean (and some snow fell at some point, so it is not trivial)
        # success = self.snow_field.is_clean and self.snow_field.total_fallen > 0
        # if success:
        #     self.event_log.emit(SnowplowEvent.ALL_CLEAN, step=self.step_count)

        ####We dont want episode termination by success only by maximun steps per episode
        success = False
        #########################

        # 9. rewards from this step's events (shared rewards are split equally between plows)
        events = self.event_log.events()
        team_of_agent = {agent.index: None for agent in self.agents}  # plows form one team
        rewards = self.reward_calculator.calculate_rewards(events, team_of_agent).per_agent()
        self.metrics.record_step(rewards, events)

        # 10b. termination (task finished) / truncation (time ran out)
        truncated_flag = (not success) and self.step_count >= self.max_steps
        self._finished = success or truncated_flag
        terminations = {agent.index: success for agent in self.agents}
        truncations = {agent.index: truncated_flag for agent in self.agents}

        # 11-12. observations + info
        observations = self.gen_obs()
        infos = {agent.index: {"movement": results[agent.index]["status"]} for agent in self.agents}
        if self._finished:
            summary = self.episode_metrics()
            for info in infos.values():
                info["episode"] = summary

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    # --------------------------------------------------------- step: helpers
    def _validate_actions(self, actions: dict) -> dict[AgentID, MovementActions]:
        """Stage 2: turn whatever the caller sent into one MovementActions per plow.

        Accepts a MovementActions member or its integer index in ACTIONS. A plow that is not in
        the dict simply waits. Anything else is a programming error and raises immediately.
        """
        checked = {}
        for agent in self.agents:
            action = actions.get(agent.index, MovementActions.DO_NOTHING)
            if isinstance(action, MovementActions):
                checked[agent.index] = action
            elif isinstance(action, (int, np.integer)) and 0 <= int(action) < len(ACTIONS):
                checked[agent.index] = ACTIONS[int(action)]
            else:
                raise ValueError(f"invalid action {action!r} for plow {agent.index}")
        return checked

    def is_move_allowed(self, grid, current_pos, intended_pos) -> bool:
        """Domain hook used by shared/movement.py: do the traffic rules allow this move?

        Wrong-way lane moves are rejected here. (Walls and map edges are already rejected by the
        shared movement code, and the rule table contains no move into them either.)
        """
        (cx, cy), (ix, iy) = tuple(current_pos), tuple(intended_pos)
        try:
            action = MovementActions((int(ix) - int(cx), int(iy) - int(cy)))
        except ValueError:  # not a one-cell step
            return False
        return bool(self.road_map.allowed[int(cy), int(cx), ACTIONS.index(action)])

    def _emit_movement_events(self, results: dict) -> None:
        """Stage 5b: describe what happened to each plow as events (the rewards read these)."""
        for agent_id, result in results.items():
            status = result["status"]
            if status == "moved":
                event_type = EventType.MOVED
            elif status == "waited":
                event_type = EventType.DO_NOTHING
            elif status == "plowing":
                event_type = SnowplowEvent.PLOWING
            elif status in _COLLISION_STATUSES:
                event_type = EventType.AGENT_COLLISION
            else:
                event_type = EventType.INVALID_MOVE
            self.event_log.emit(
                event_type,
                agent_id=agent_id,
                step=self.step_count,
                status=status,
                current_position=tuple(int(v) for v in result["current_position"]),
                resolved_position=tuple(int(v) for v in result["resolved_position"]),
            )
            # Draw the plow facing the way it drove.
            if status == "moved":
                self.agents[agent_id].state.dir = _FACING[
                    MovementActions(tuple(int(b) - int(a) for a, b in zip(result["current_position"], result["resolved_position"])))
                ]

    def _plow_snow(self, results: dict) -> None:
        """Stage 6: time-based, cooperative plowing.

        1. Count how many plows stand on every cell after the movement stage (cells can be shared).
        2. Every snowy cell with plows on it gets one step of work: 1 / plow_time(level, n_plows)
           (see SnowField.plow_step). Two plows on a level-3 cell finish in 3 steps instead of 8.
        3. Remember how much of the work each plow did on each cell (self._work_share).
        4. When a cell becomes clean, every plow that worked on it gets a PARTIALLY_CLEARED event
           whose reward_scale is (layers removed) x (its share of the work). The shares add up to 1,
           so the team reward of a cell is the same whoever cleaned it.
        5. Only after all cells are processed, check which streets became completely snow-free,
           so the result does not depend on the order of the plows.
        """
        snow = self.snow_field.snow
        plows_on_cell: dict[tuple[int, int], list[AgentID]] = {}
        for agent_id, result in results.items():
            pos = tuple(int(v) for v in result["resolved_position"])
            plows_on_cell.setdefault(pos, []).append(agent_id)

        # Bookkeeping that does not change the world.
        for pos, agent_ids in plows_on_cell.items():
            if len(agent_ids) > 1:
                self.metrics.add("shared_cell_steps", len(agent_ids))  # plow-steps spent sharing a cell
        for agent_id, result in results.items():
            x, y = (int(v) for v in result["resolved_position"])
            if result["status"] == "moved" and snow[y, x] == 0:
                self.metrics.add("repeated_work")  # drove onto a cell that was already clean
            if snow[y, x] > 0:
                self.metrics.add("plowing_steps")  # one plow-step of plowing work

        # Work: who contributes how much to which cell (before plow_step changes the snow).
        workers = {pos: len(ids) for pos, ids in plows_on_cell.items() if snow[pos[1], pos[0]] > 0}
        for pos, n_plows in workers.items():
            step_work = 1.0 / self.config.snow.plow_time(int(snow[pos[1], pos[0]]), n_plows)
            shares = self._work_share.setdefault(pos, {})
            for agent_id in plows_on_cell[pos]:
                shares[agent_id] = shares.get(agent_id, 0.0) + step_work / n_plows

        cleared = self.snow_field.plow_step(workers)

        contributors: dict[int, set[AgentID]] = {}  # street id -> plows that cleaned a cell on it
        for (x, y), removed in cleared.items():
            shares = self._work_share.pop((x, y), {})
            total = sum(shares.values()) or 1.0
            self.metrics.add("snow_cleared", removed)
            self.metrics.add("cells_cleared")
            if len(shares) > 1:
                self.metrics.add("cells_cleared_together")  # the cooperation we want the plows to learn
            for agent_id, share in shares.items():
                self.event_log.emit(
                    SnowplowEvent.PARTIALLY_CLEARED,
                    agent_id=agent_id,
                    step=self.step_count,
                    position=(x, y),
                    removed=removed,
                    n_plows=len(shares),
                    reward_scale=removed * share / total,
                )
            street = int(self.road_map.street_id[y, x])
            contributors.setdefault(street, set()).update(shares)

        for street, agent_ids in contributors.items():
            if self.snow_field.street_snow(street) > 0:
                continue  # not completely clean yet
            is_main = bool(self.road_map.street_is_main[street])
            event = SnowplowEvent.MAIN_STREET_CLEARED if is_main else SnowplowEvent.SECONDARY_STREET_CLEARED
            self.metrics.add("main_streets_cleared" if is_main else "secondary_streets_cleared")
            for agent_id in sorted(agent_ids):  # every plow that helped gets the street reward
                self.event_log.emit(event, agent_id=agent_id, step=self.step_count, street=street)

    # --------------------------------------------------------------- metrics
    def episode_metrics(self) -> dict:
        """Metrics of the episode so far (steps, reward, snow cleared, % clean, success, ...)."""
        summary = self.metrics.summary()
        for key in (
            "snow_cleared", "repeated_work", "main_streets_cleared", "secondary_streets_cleared",
            "cells_cleared", "cells_cleared_together", "plowing_steps", "shared_cell_steps",
        ):
            summary.setdefault(key, 0.0)
        for key in ("events/moved", "events/do_nothing", "events/invalid_move", "events/agent_collision", "events/plowing"):
            summary.setdefault(key, 0)
        # Fraction of the cleaned cells that were cleaned by 2+ plows together (cooperation indicator).
        summary["cooperation_rate"] = summary["cells_cleared_together"] / summary["cells_cleared"] if summary["cells_cleared"] else 0.0
        summary["snow_fallen"] = self.snow_field.total_fallen
        summary["snow_remaining"] = int(self.snow_field.snow.sum())
        summary["fraction_clean"] = self.snow_field.fraction_clean
        # summary["success"] = bool(self.snow_field.is_clean and self.snow_field.total_fallen > 0)
        return summary

    # ------------------------------------------------------------- rendering
    def get_full_render(self, highlight: bool, tile_size: int):
        """Draw the map. Cells are refreshed from the numpy state first (see snowplow/cell.py)."""
        snow = self.snow_field.snow
        progress = self.snow_field.plow_progress
        clouds = self.snow_field.cloud_mask()
        plows_here = np.zeros(snow.shape, dtype=int)  # plows per cell, to mark team plowing
        for agent in self.agents:
            x, y = (int(v) for v in agent.state.pos)
            plows_here[y, x] += 1
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid.get(x, y)
                if cell.can_overlap():
                    cell.set_appearance(int(snow[y, x]), bool(clouds[y, x]), float(progress[y, x]), plows_here[y, x] > 1)
        return self.grid.render(tile_size, agents=self.agents, highlight_mask=None)
