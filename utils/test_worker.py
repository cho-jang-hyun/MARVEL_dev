import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrowPatch, Rectangle
from skimage.draw import polygon as sk_polygon
from skimage.morphology import label
from collections import deque
import gc
import time

from utils.env import Env
from utils.agent import Agent
from utils.utils import *
from utils.node_manager import NodeManager
from utils.motion_model import compute_allowable_heading
from utils.runtime_config import *

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


def load_compatible_state_dict(module, checkpoint_state, label):
    model_state = module.state_dict()
    compatible_state = {
        key: value for key, value in checkpoint_state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    missing_or_mismatched = len(model_state) - len(compatible_state)
    module.load_state_dict(compatible_state, strict=False)
    print(f'Loaded {label}: {len(compatible_state)} tensors, skipped {missing_or_mismatched}')

class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, n_agent, fov, sensor_range, utility_range,
                 budget_timesteps=None, device='cpu', save_image=False, greedy=True):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.fov = fov
        self.sim_steps = NUM_SIM_STEPS
        self.sensor_range = sensor_range
        self.greedy = greedy
        self.n_agents = n_agent
        self.scaling = 0.04

        self.env = Env(
            global_step,
            self.fov,
            self.sensor_range,
            plot=self.save_image,
            n_agents=self.n_agents,
            test_set=TEST_SET,
        )

        # Create independent node managers for each agent to ensure decentralized testing
        self.robot_list = []
        for i in range(self.n_agents):
            # Each agent gets its own independent node_manager for decentralized testing
            individual_node_manager = NodeManager(self.fov, self.sensor_range, utility_range, plot=self.save_image)

            agent = Agent(i, policy_net, self.fov, self.env.angles[i], self.sensor_range,
                         individual_node_manager, None, self.device, self.save_image)
            self.robot_list.append(agent)

        self.perf_metrics = dict()

        # Initialize trajectory buffer for each agent
        self.trajectory_buffer = {}
        for i in range(self.n_agents):
            self.trajectory_buffer[i] = deque(maxlen=TRAJECTORY_HISTORY_LENGTH)
            # Initialize with starting positions (x, y, heading, velocity=0)
            start_location = self.env.robot_locations[i]
            self.trajectory_buffer[i].append((
                start_location[0],
                start_location[1],
                self.env.angles[i],
                0.0
            ))
        self.base_locations = self.env.robot_locations.copy()
        if budget_timesteps is None:
            budget_meters = float(BUDGET)
        else:
            budget_meters = float(budget_timesteps) * float(BUDGET_TIMESTEP_METERS)
        self.initial_budgets = np.full(self.n_agents, budget_meters, dtype=float)
        self.remaining_budgets = self.initial_budgets.copy()
        self.returning_agents = np.zeros(self.n_agents, dtype=bool)
        self.broken_agents = np.zeros(self.n_agents, dtype=bool)
        self.breakdown_enabled = bool(globals().get('TEST_BREAKDOWN_STUDY_ENABLED', False))
        self.breakdown_agent_count = min(
            max(int(globals().get('TEST_BREAKDOWN_AGENT_COUNT', 0)), 0),
            self.n_agents,
        )
        self.breakdown_random_seed = int(globals().get('TEST_BREAKDOWN_RANDOM_SEED', 0))
        self.breakdown_episode_seed = self.breakdown_random_seed + int(self.global_step)
        self.breakdown_scheduled_steps = {}
        self.breakdown_triggered_steps = {}
        self._schedule_breakdowns()
        self.merged_objective_completed = False
        self.merged_completion_travel_dist = None
        self.merged_total_utility = np.nan
        self.total_free_cells = int(np.count_nonzero(self.env.ground_truth == FREE))
        self._sensing_mask_shape = self.env.ground_truth.shape
        self._sensor_range_cells = float(self.sensor_range / CELL_SIZE)
        self._sector_offsets_cache = {}
        self.individual_maps = {}
        if self.save_image:
            for i in range(self.n_agents):
                self.individual_maps[i] = self.env.get_agent_map_info(i).map.copy()

    def _schedule_breakdowns(self):
        if not self.breakdown_enabled or self.breakdown_agent_count == 0:
            return

        min_step = int(globals().get('TEST_BREAKDOWN_MIN_STEP', 0))
        max_step = int(globals().get('TEST_BREAKDOWN_MAX_STEP', MAX_EPISODE_STEP))
        min_step = max(min_step, 0)
        max_step = max(max_step, min_step)

        rng = np.random.default_rng(self.breakdown_episode_seed)
        agent_ids = np.sort(rng.choice(self.n_agents, size=self.breakdown_agent_count, replace=False))
        breakdown_steps = rng.integers(min_step, max_step + 1, size=self.breakdown_agent_count)
        self.breakdown_scheduled_steps = {
            int(agent_id): int(step)
            for agent_id, step in zip(agent_ids, breakdown_steps)
        }

    def _active_agent_ids_for_merged_map(self):
        return [agent_id for agent_id in range(self.n_agents) if not self.broken_agents[agent_id]]

    def _refresh_merged_belief_excluding_broken(self):
        active_agent_ids = self._active_agent_ids_for_merged_map()
        merged_belief = np.ones(self.env.ground_truth_size) * UNKNOWN
        if len(active_agent_ids) > 0:
            occupied_mask = np.zeros(self.env.ground_truth_size, dtype=bool)
            free_mask = np.zeros(self.env.ground_truth_size, dtype=bool)
            for agent_id in active_agent_ids:
                belief = self.env.agent_beliefs[agent_id]
                occupied_mask |= belief == OCCUPIED
                free_mask |= belief == FREE
            merged_belief[free_mask] = FREE
            merged_belief[occupied_mask] = OCCUPIED

        self.env.robot_belief = merged_belief
        self.env.belief_info.update_map_info(
            self.env.robot_belief,
            self.env.belief_origin_x,
            self.env.belief_origin_y,
        )
        self.env.evaluate_exploration_rate()

    def _trigger_scheduled_breakdowns(self, decision_step):
        triggered_agent_ids = []
        if not self.breakdown_enabled:
            return triggered_agent_ids

        for agent_id, scheduled_step in self.breakdown_scheduled_steps.items():
            if self.broken_agents[agent_id] or scheduled_step != decision_step:
                continue
            self.broken_agents[agent_id] = True
            self.returning_agents[agent_id] = False
            self.breakdown_triggered_steps[int(agent_id)] = int(decision_step)
            triggered_agent_ids.append(int(agent_id))

        if triggered_agent_ids:
            self._refresh_merged_belief_excluding_broken()

        return triggered_agent_ids

    def _plot_current_breakdown_status_frame(self, decision_step):
        if not self.save_image:
            return
        robot_locations = [
            get_cell_position_from_coords(robot.location, self.env.belief_info)
            for robot in self.robot_list
        ]
        robot_headings = [robot.heading for robot in self.robot_list]
        self.plot_local_env_sim(f'{decision_step}_breakdown', robot_locations, robot_headings)

    def _refresh_merged_belief_after_updates(self):
        if self.breakdown_enabled and np.any(self.broken_agents):
            self._refresh_merged_belief_excluding_broken()
        else:
            self.env.refresh_merged_belief()

    def _non_broken_agents_all_returning(self):
        non_broken_agents = ~self.broken_agents
        return bool(np.any(non_broken_agents) and np.all(self.returning_agents[non_broken_agents]))

    def _all_agents_broken(self):
        return bool(np.all(self.broken_agents))

    def _non_broken_agents_at_base(self):
        return all(
            np.allclose(robot.location, self.base_locations[robot.id])
            for robot in self.robot_list
            if not self.broken_agents[robot.id]
        )

    def _plot_breakdown_label(self, location):
        plt.text(
            location[0] + 4,
            location[1] - 4,
            'broken',
            color='white',
            fontsize=8,
            fontweight='bold',
            bbox=dict(facecolor='black', edgecolor='red', boxstyle='round,pad=0.2', alpha=0.85),
            zorder=20,
            clip_on=False,
        )

    def _update_merged_graph(self):
        # Test-time evaluation keeps merged map visualization from env beliefs only.
        # Merged-node graph maintenance is intentionally skipped.
        self.merged_total_utility = np.nan

    def _get_max_travel_dist(self):
        return max((robot.travel_dist for robot in self.robot_list), default=0.0)

    def _get_merged_explored_rate(self):
        if self.total_free_cells <= 0:
            return 0.0
        return float(np.count_nonzero(self.env.robot_belief == FREE) / self.total_free_cells)

    def _get_remaining_budget_summary(self):
        remaining = np.asarray(self.remaining_budgets, dtype=float)
        return (
            float(np.mean(remaining)),
            float(np.min(remaining)),
            float(np.max(remaining)),
        )

    def _update_objective_state(self):
        merged_completed_this_step = (
            not self.merged_objective_completed
            and self.env.explored_rate >= SUCCESS_THRESHOLD
        )
        if merged_completed_this_step:
            self.merged_objective_completed = True
            self.merged_completion_travel_dist = self._get_max_travel_dist()
        return merged_completed_this_step

    def _local_objective_completed(self, robot):
        if self.total_free_cells <= 0:
            return False
        agent_map = self.env.agent_beliefs[robot.id]
        agent_explored_rate = np.count_nonzero(agent_map == FREE) / self.total_free_cells
        return bool(agent_explored_rate >= SUCCESS_THRESHOLD)

    def _set_agent_budget_context(self, robot):
        robot.set_budget_context(
            self.initial_budgets[robot.id],
            self.remaining_budgets[robot.id],
            self.base_locations[robot.id],
            return_mode=self.returning_agents[robot.id],
        )

    def _get_return_path(self, robot):
        self._set_agent_budget_context(robot)
        return robot.get_path_to_base()

    @staticmethod
    def _location_key(location):
        return float(location[0]), float(location[1])

    def _get_collision_free_candidate(self, robot, occupied_keys):
        for _, _, candidate_location, candidate_heading_index in robot.current_waypoint_candidates:
            candidate_key = self._location_key(candidate_location)
            if candidate_key in occupied_keys:
                continue
            return candidate_location.copy(), candidate_heading_index

        return None, None

    def _has_feasible_exploration_action(self, robot, observation):
        current_edge = observation[4][0, :, 0].detach().cpu().numpy()
        edge_padding = observation[5][0, 0].detach().cpu().numpy()
        valid_slots = np.where(edge_padding == 0)[0]
        for slot in valid_slots:
            if int(current_edge[slot]) == int(robot.current_index):
                continue
            return True
        return False

    def _get_sector_offsets(self, heading):
        heading_key = float(heading % 360)
        cached_offsets = self._sector_offsets_cache.get(heading_key)
        if cached_offsets is not None:
            return cached_offsets

        start_angle = (heading_key - self.fov / 2 + 360) % 360
        end_angle = (heading_key + self.fov / 2) % 360
        if start_angle <= end_angle:
            angle_range = np.linspace(start_angle, end_angle, 20)
        else:
            angle_range = np.concatenate([
                np.linspace(start_angle, 360, 10),
                np.linspace(0, end_angle, 10),
            ])

        angle_radians = np.radians(angle_range)
        x_offsets = self._sensor_range_cells * np.cos(angle_radians)
        y_offsets = self._sensor_range_cells * np.sin(angle_radians)
        self._sector_offsets_cache[heading_key] = (x_offsets, y_offsets)
        return x_offsets, y_offsets

    def _get_heading_index_towards(self, robot, next_location):
        next_location = np.asarray(next_location, dtype=float)
        if np.allclose(next_location, robot.location):
            return int((robot.heading % 360) / 360 * NUM_ANGLES_BIN) % NUM_ANGLES_BIN
        angle = np.degrees(np.arctan2(
            next_location[1] - robot.location[1],
            next_location[0] - robot.location[0],
        ) % (2 * np.pi))
        return int(angle / 360 * NUM_ANGLES_BIN) % NUM_ANGLES_BIN

    def _append_trajectory_step(self, robot, next_location):
        prev_trajectory = self.trajectory_buffer[robot.id][-1] if len(self.trajectory_buffer[robot.id]) > 0 else None
        if prev_trajectory is not None:
            prev_x, prev_y = prev_trajectory[0], prev_trajectory[1]
            velocity = np.linalg.norm(next_location - np.array([prev_x, prev_y])) / NUM_SIM_STEPS
        else:
            velocity = 0.0

        self.trajectory_buffer[robot.id].append((
            next_location[0],
            next_location[1],
            robot.heading,
            velocity
        ))

    def _all_agents_at_base(self):
        return all(np.allclose(robot.location, self.base_locations[robot.id]) for robot in self.robot_list)

    @staticmethod
    def _should_force_return(distance_to_base, remaining_budget):
        return distance_to_base + RETURN_SAFETY_MARGIN >= remaining_budget

    def _exploration_move_preserves_return_budget(self, robot, next_location):
        step_distance = float(np.linalg.norm(np.asarray(next_location, dtype=float) - robot.location))
        next_distance_to_base = robot.get_distance_to_base(start_location=next_location)
        if not np.isfinite(next_distance_to_base):
            return False
        required_budget = step_distance + next_distance_to_base + RETURN_SAFETY_MARGIN
        return required_budget <= self.remaining_budgets[robot.id] + 1e-6

    def _set_return_target(self, robot, selected_locations, dist_list, next_heading_index_list):
        robot_at_base = np.allclose(robot.location, self.base_locations[robot.id])
        if robot_at_base:
            selected_locations[robot.id] = robot.location.copy()
            dist_list[robot.id] = 0.0
            next_heading_index_list[robot.id] = self._get_heading_index_towards(robot, robot.location)
            return False

        return_path = self._get_return_path(robot)
        if len(return_path) == 0:
            selected_locations[robot.id] = robot.location.copy()
            dist_list[robot.id] = 0.0
            next_heading_index_list[robot.id] = self._get_heading_index_towards(robot, robot.location)
            return True

        selected_locations[robot.id] = return_path[0].copy()
        dist_list[robot.id] = np.linalg.norm(selected_locations[robot.id] - robot.location)
        next_heading_index_list[robot.id] = self._get_heading_index_towards(robot, selected_locations[robot.id])
        return False

    @staticmethod
    def _budget_to_decision_steps(budget_value):
        if BUDGET_TIMESTEP_METERS <= 0:
            return 0
        return int(np.ceil(max(float(budget_value), 0.0) / BUDGET_TIMESTEP_METERS))

    def run_episode(self):
        for robot in self.robot_list:
            robot.update_graph(self.env.get_agent_map_info(robot.id), self.env.robot_locations[robot.id].copy())
        for robot in self.robot_list:
            robot.update_planning_state()
            self._set_agent_budget_context(robot)
        self._update_merged_graph()
        self._update_objective_state()

        merged_reached_0_90 = False
        merged_reached_0_99 = False
        merged_dist_to_0_90 = np.nan
        merged_dist_to_0_99 = np.nan

        individual_reached_0_90 = np.zeros(self.n_agents, dtype=bool)
        individual_reached_0_99 = np.zeros(self.n_agents, dtype=bool)
        individual_dist_to_0_90 = np.full(self.n_agents, np.nan, dtype=float)
        individual_dist_to_0_99 = np.full(self.n_agents, np.nan, dtype=float)

        compute_time_history = []
        mission_failure = False
        max_decision_steps = MAX_EPISODE_STEP + self._budget_to_decision_steps(np.max(self.initial_budgets))

        length_history = [self._get_max_travel_dist()]
        explored_rate_history = [self.env.explored_rate]
        merged_explored_rate_history = [self._get_merged_explored_rate()]
        individual_explored_rate_history = [self._get_agent_explored_rates()]
        remaining_budget_history = [self.remaining_budgets.astype(float).tolist()]
        remaining_budget_mean_history = []
        remaining_budget_min_history = []
        remaining_budget_max_history = []
        remaining_mean, remaining_min, remaining_max = self._get_remaining_budget_summary()
        remaining_budget_mean_history.append(remaining_mean)
        remaining_budget_min_history.append(remaining_min)
        remaining_budget_max_history.append(remaining_max)
        overlap_rate = self.compute_overlap_rate(self.env.robot_locations, self.env.angles)
        overlap_ratio_history = [overlap_rate]

        for i in range(max_decision_steps):
            step_start_time = time.time()
            triggered_agent_ids = self._trigger_scheduled_breakdowns(i)
            if triggered_agent_ids:
                self._plot_current_breakdown_status_frame(i)
            selected_locations = [robot.location.copy() for robot in self.robot_list]
            dist_list = [0.0 for _ in range(self.n_agents)]
            next_heading_index_list = [self._get_heading_index_towards(robot, robot.location) for robot in self.robot_list]
            observations = {}
            active_explorer_ids = []

            for robot in self.robot_list:
                if self.broken_agents[robot.id]:
                    continue
                self._set_agent_budget_context(robot)
                robot_at_base = np.allclose(robot.location, self.base_locations[robot.id])
                distance_to_base = robot.get_distance_to_base()

                if self.returning_agents[robot.id]:
                    if robot_at_base:
                        continue
                    return_path = self._get_return_path(robot)
                    if len(return_path) == 0:
                        if not robot_at_base:
                            mission_failure = True
                        selected_locations[robot.id] = robot.location.copy()
                    else:
                        selected_locations[robot.id] = return_path[0].copy()
                    dist_list[robot.id] = np.linalg.norm(selected_locations[robot.id] - robot.location)
                    next_heading_index_list[robot.id] = self._get_heading_index_towards(robot, selected_locations[robot.id])
                    continue

                if (
                    self._local_objective_completed(robot)
                    or self._should_force_return(distance_to_base, self.remaining_budgets[robot.id])
                ):
                    self.returning_agents[robot.id] = True
                    return_path = self._get_return_path(robot)
                    if len(return_path) == 0:
                        if not robot_at_base:
                            mission_failure = True
                        selected_locations[robot.id] = robot.location.copy()
                    else:
                        selected_locations[robot.id] = return_path[0].copy()
                    dist_list[robot.id] = np.linalg.norm(selected_locations[robot.id] - robot.location)
                    next_heading_index_list[robot.id] = self._get_heading_index_towards(robot, selected_locations[robot.id])
                    continue

                observation = robot.get_observation(
                    robot_locations=self.env.robot_locations,
                    trajectory_buffer=self.trajectory_buffer
                )

                if not self._has_feasible_exploration_action(robot, observation):
                    self.returning_agents[robot.id] = True
                    return_path = self._get_return_path(robot)
                    if len(return_path) == 0:
                        if not robot_at_base:
                            mission_failure = True
                        selected_locations[robot.id] = robot.location.copy()
                    else:
                        selected_locations[robot.id] = return_path[0].copy()
                    dist_list[robot.id] = np.linalg.norm(selected_locations[robot.id] - robot.location)
                    next_heading_index_list[robot.id] = self._get_heading_index_towards(robot, selected_locations[robot.id])
                    continue

                observations[robot.id] = observation
                active_explorer_ids.append(robot.id)

            if mission_failure:
                break

            if len(active_explorer_ids) == 0 and self._non_broken_agents_all_returning():
                if not SIMULATE_RETURN_TO_BASE or self._non_broken_agents_at_base():
                    break
            if len(active_explorer_ids) == 0 and self._all_agents_broken():
                break

            candidate_explorer_ids = active_explorer_ids
            active_explorer_ids = []
            for robot_id in candidate_explorer_ids:
                robot = self.robot_list[robot_id]
                observation = observations[robot_id]
                next_location, _, _, next_heading_index = robot.select_next_waypoint(observation, greedy=self.greedy)
                if not self._exploration_move_preserves_return_budget(robot, next_location):
                    self.returning_agents[robot_id] = True
                    if self._set_return_target(robot, selected_locations, dist_list, next_heading_index_list):
                        mission_failure = True
                    continue

                selected_locations[robot_id] = next_location.copy()
                dist_list[robot_id] = np.linalg.norm(next_location - robot.location)
                next_heading_index_list[robot_id] = next_heading_index
                active_explorer_ids.append(robot_id)

            if mission_failure:
                break

            if len(active_explorer_ids) == 0 and self._non_broken_agents_all_returning():
                if not SIMULATE_RETURN_TO_BASE or self._non_broken_agents_at_base():
                    break
            if len(active_explorer_ids) == 0 and self._all_agents_broken():
                break

            selected_locations = np.asarray(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.asarray(dist_list))
            selected_locations_in_arriving_sequence = selected_locations[arriving_sequence].copy()
            active_explorer_set = set(active_explorer_ids)
            resolved_location_keys = set()

            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                selected_key = self._location_key(selected_location)
                if selected_key in resolved_location_keys:
                    robot_id = arriving_sequence[j]
                    if robot_id in active_explorer_set:
                        replacement_location, replacement_heading_index = self._get_collision_free_candidate(
                            self.robot_list[robot_id],
                            resolved_location_keys,
                        )
                        if replacement_location is None:
                            replacement_location = self.robot_list[robot_id].location.copy()
                            replacement_heading_index = self._get_heading_index_towards(self.robot_list[robot_id], replacement_location)
                    else:
                        replacement_location = self.robot_list[robot_id].location.copy()
                        replacement_heading_index = self._get_heading_index_towards(self.robot_list[robot_id], replacement_location)

                    selected_location = replacement_location
                    selected_key = self._location_key(selected_location)
                    next_heading_index_list[robot_id] = replacement_heading_index
                    selected_locations_in_arriving_sequence[j] = selected_location
                    selected_locations[robot_id] = selected_location
                resolved_location_keys.add(selected_key)

            still_active_explorer_ids = []
            selected_locations_list = [selected_locations[k].copy() for k in range(self.n_agents)]
            for robot_id in active_explorer_ids:
                robot = self.robot_list[robot_id]
                if self._exploration_move_preserves_return_budget(robot, selected_locations[robot_id]):
                    still_active_explorer_ids.append(robot_id)
                    continue

                self.returning_agents[robot_id] = True
                if self._set_return_target(robot, selected_locations_list, dist_list, next_heading_index_list):
                    mission_failure = True

            if mission_failure:
                break

            active_explorer_ids = still_active_explorer_ids
            selected_locations = np.asarray(selected_locations_list).reshape(-1, 2)

            if len(active_explorer_ids) == 0 and self._non_broken_agents_all_returning():
                if not SIMULATE_RETURN_TO_BASE or self._non_broken_agents_at_base():
                    break
            if len(active_explorer_ids) == 0 and self._all_agents_broken():
                break

            robot_locations_sim = []
            robot_headings_sim = []
            all_robots_heading_list = []
            for robot, next_location, next_heading_index in zip(self.robot_list, selected_locations, next_heading_index_list):
                if self.broken_agents[robot.id]:
                    robot_cell = get_cell_position_from_coords(robot.location, self.env.belief_info)
                    robot_locations_sim.append(np.tile(robot_cell, (self.sim_steps, 1)))
                    robot_headings_sim.append([robot.heading for _ in range(self.sim_steps)])
                    all_robots_heading_list.append(robot.heading)
                    continue

                robot_current_cell = get_cell_position_from_coords(robot.location, self.env.belief_info)
                robot_cell = get_cell_position_from_coords(next_location, self.env.belief_info)

                next_heading = next_heading_index * (360 / NUM_ANGLES_BIN)
                final_heading = compute_allowable_heading(
                    robot.location, next_location, robot.heading, next_heading, robot.velocity, robot.yaw_rate
                )

                intermediate_cells = np.linspace(robot_current_cell, robot_cell, self.sim_steps + 1)[1:]
                intermediate_cells = np.round(intermediate_cells).astype(int)
                intermediate_headings = self.smooth_heading_change(robot.heading, final_heading, steps=self.sim_steps)

                robot_locations_sim.append(intermediate_cells)
                robot_headings_sim.append(intermediate_headings)
                all_robots_heading_list.append(final_heading)
                robot.update_heading(final_heading)

            for l in range(self.sim_steps):
                robot_location_sim_step = []
                robot_heading_sim_step = []
                for q in range(self.n_agents):
                    if not self.broken_agents[q]:
                        self.env.update_robot_belief(
                            q,
                            robot_locations_sim[q][l],
                            robot_headings_sim[q][l],
                            refresh_merged=False,
                        )
                    if self.save_image and not self.broken_agents[q]:
                        self.individual_maps[q] = self.env.get_agent_map_info(q).map.copy()
                    robot_location_sim_step.append(robot_locations_sim[q][l])
                    robot_heading_sim_step.append(robot_headings_sim[q][l])
                self._refresh_merged_belief_after_updates()

                if self.save_image:
                    num_frame = i * self.sim_steps + l
                    self.plot_local_env_sim(num_frame, robot_location_sim_step, robot_heading_sim_step)
                    if num_frame % 5 == 0:
                        self.plot_individual_agent_views(num_frame, robot_location_sim_step, robot_heading_sim_step)

            previous_locations = [robot.location.copy() for robot in self.robot_list]
            for robot, next_location in zip(self.robot_list, selected_locations):
                if self.broken_agents[robot.id]:
                    continue
                self.env.final_sim_step(next_location, robot.id)
                traveled_distance = float(np.linalg.norm(previous_locations[robot.id] - next_location))
                if traveled_distance > 0.0:
                    self.remaining_budgets[robot.id] -= traveled_distance
                self._append_trajectory_step(robot, next_location)

            for robot in self.robot_list:
                if self.broken_agents[robot.id]:
                    continue
                robot.update_graph(self.env.get_agent_map_info(robot.id), self.env.robot_locations[robot.id].copy())
            for robot in self.robot_list:
                if self.broken_agents[robot.id]:
                    continue
                robot.mark_nodes_visited_by_others(self.env.robot_locations, self.trajectory_buffer)
                robot.update_planning_state()
                self._set_agent_budget_context(robot)
                
            compute_time_history.append(time.time() - step_start_time)

            self._update_merged_graph()
            merged_completed_this_step = self._update_objective_state()
            overlap_rate = self.compute_overlap_rate(selected_locations, all_robots_heading_list)

            max_travel_dist = self._get_max_travel_dist()
            length_history.append(max_travel_dist)
            explored_rate_history.append(self.env.explored_rate)
            merged_explored_rate_history.append(self._get_merged_explored_rate())
            individual_explored_rates = self._get_agent_explored_rates()
            individual_explored_rate_history.append(individual_explored_rates)
            remaining_budget_history.append(self.remaining_budgets.astype(float).tolist())
            remaining_mean, remaining_min, remaining_max = self._get_remaining_budget_summary()
            remaining_budget_mean_history.append(remaining_mean)
            remaining_budget_min_history.append(remaining_min)
            remaining_budget_max_history.append(remaining_max)
            overlap_ratio_history.append(overlap_rate)
            if (self.env.explored_rate >= 0.90) and (not merged_reached_0_90):
                merged_dist_to_0_90 = max_travel_dist
                merged_reached_0_90 = True
            if (self.env.explored_rate >= 0.99) and (not merged_reached_0_99):
                merged_dist_to_0_99 = max_travel_dist
                merged_reached_0_99 = True

            for agent_id, agent_rate in enumerate(individual_explored_rates):
                if (agent_rate >= 0.90) and (not individual_reached_0_90[agent_id]):
                    individual_dist_to_0_90[agent_id] = self.robot_list[agent_id].travel_dist
                    individual_reached_0_90[agent_id] = True
                if (agent_rate >= 0.99) and (not individual_reached_0_99[agent_id]):
                    individual_dist_to_0_99[agent_id] = self.robot_list[agent_id].travel_dist
                    individual_reached_0_99[agent_id] = True

            if merged_completed_this_step:
                self.merged_completion_travel_dist = self._get_max_travel_dist()
            if np.any(self.remaining_budgets < 0):
                mission_failure = True

            if mission_failure:
                break
            if self._non_broken_agents_all_returning():
                if not SIMULATE_RETURN_TO_BASE or self._non_broken_agents_at_base():
                    break
            if self._all_agents_broken():
                break

        # Save metrics
        final_travel_dist = self._get_max_travel_dist()
        self.perf_metrics['travel_dist'] = final_travel_dist
        self.perf_metrics['merged_travel_dist'] = (
            self.merged_completion_travel_dist
            if self.merged_completion_travel_dist is not None else final_travel_dist
        )
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = bool(self.merged_objective_completed)
        self.perf_metrics['dist_to_0_90'] = merged_dist_to_0_90
        self.perf_metrics['dist_to_0_99'] = merged_dist_to_0_99
        final_individual_rates = np.asarray(self._get_agent_explored_rates(), dtype=float)
        self.perf_metrics['individual_explored_rates'] = final_individual_rates.tolist()
        self.perf_metrics['individual_explored_rate_mean'] = float(np.mean(final_individual_rates))
        self.perf_metrics['individual_explored_rate_std'] = float(np.std(final_individual_rates))
        self.perf_metrics['individual_dist_to_0_90'] = individual_dist_to_0_90.tolist()
        self.perf_metrics['individual_dist_to_0_99'] = individual_dist_to_0_99.tolist()
        self.perf_metrics['compute_time_history'] = compute_time_history
        if len(compute_time_history) > 0:
            compute_time_array = np.asarray(compute_time_history, dtype=float)
            self.perf_metrics['compute_time_mean'] = float(np.mean(compute_time_array))
            self.perf_metrics['compute_time_std'] = float(np.std(compute_time_array))
        else:
            self.perf_metrics['compute_time_mean'] = np.nan
            self.perf_metrics['compute_time_std'] = np.nan
        self.perf_metrics['length_history'] = length_history
        self.perf_metrics['explored_rate_history'] = explored_rate_history
        self.perf_metrics['merged_explored_rate_history'] = merged_explored_rate_history
        self.perf_metrics['individual_explored_rate_history'] = individual_explored_rate_history
        self.perf_metrics['initial_budget_history'] = self.initial_budgets.astype(float).tolist()
        self.perf_metrics['remaining_budget_history'] = remaining_budget_history
        self.perf_metrics['remaining_budget_mean_history'] = remaining_budget_mean_history
        self.perf_metrics['remaining_budget_min_history'] = remaining_budget_min_history
        self.perf_metrics['remaining_budget_max_history'] = remaining_budget_max_history
        self.perf_metrics['overlap_ratio_history'] = overlap_ratio_history
        self.perf_metrics['breakdown_enabled'] = bool(self.breakdown_enabled)
        self.perf_metrics['breakdown_agent_count'] = int(self.breakdown_agent_count)
        self.perf_metrics['breakdown_random_seed'] = int(self.breakdown_random_seed)
        self.perf_metrics['broken_agent_ids'] = [
            int(agent_id) for agent_id in np.where(self.broken_agents)[0]
        ]
        self.perf_metrics['breakdown_scheduled_steps'] = {
            int(agent_id): int(step)
            for agent_id, step in self.breakdown_scheduled_steps.items()
        }
        self.perf_metrics['breakdown_triggered_steps'] = {
            int(agent_id): int(step)
            for agent_id, step in self.breakdown_triggered_steps.items()
        }
        self.perf_metrics['num_broken_agents_final'] = int(np.count_nonzero(self.broken_agents))
    
        # Save episode video.
        if self.save_image:
            pass
            make_video_test(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate, self.n_agents, self.fov, self.sensor_range)

    def smooth_heading_change(self, prev_heading, heading, steps=10):
        prev_heading = prev_heading % 360
        heading = heading % 360
        diff = heading - prev_heading
        
        if abs(diff) > 180:
            diff = diff - 360 if diff > 0 else diff + 360

        intermediate_headings = [
            (prev_heading + i * diff / steps) % 360
            for i in range(1, steps)
        ]

        intermediate_headings.append(heading)
        return intermediate_headings
            
    def heading_to_vector(self, heading, length=25):
        # Convert heading to vector
        if isinstance(heading, (list, np.ndarray)):
            heading = heading[0]
        heading_rad = np.radians(heading)
        return np.cos(heading_rad) * length, np.sin(heading_rad) * length
    
    def create_sensing_mask(self, location, heading, labeled_free=None):
        mask = np.zeros(self._sensing_mask_shape, dtype=np.uint8)
        location_cell = get_cell_position_from_coords(location, self.env.belief_info)
        x_offsets, y_offsets = self._get_sector_offsets(heading)
        x_coords = np.rint(np.concatenate(([location_cell[0]], location_cell[0] + x_offsets, [location_cell[0]]))).astype(int)
        y_coords = np.rint(np.concatenate(([location_cell[1]], location_cell[1] + y_offsets, [location_cell[1]]))).astype(int)
        rr, cc = sk_polygon(y_coords, x_coords, shape=mask.shape)

        if labeled_free is None:
            free_connected_map = get_free_and_connected_map(location, self.env.belief_info)
            robot_component = free_connected_map[location_cell[1], location_cell[0]]
            mask[rr, cc] = (free_connected_map[rr, cc] == robot_component)
            return mask

        robot_component = labeled_free[location_cell[1], location_cell[0]]
        if robot_component == 0:
            return mask
        mask[rr, cc] = (labeled_free[rr, cc] == robot_component)
       
        return mask
    
    def compute_overlap_rate(self, all_robots_locations, robot_headings_list):
        total_mask = np.zeros(self._sensing_mask_shape, dtype=np.uint16)
        labeled_free = label((self.env.robot_belief == FREE).astype(np.uint8), connectivity=2)
        for robot_location, robot_heading in zip(all_robots_locations, robot_headings_list):
            total_mask += self.create_sensing_mask(robot_location, robot_heading, labeled_free=labeled_free)

        total_sensing_area = np.count_nonzero(total_mask)
        if total_sensing_area == 0:
            return 0.0
        total_overlap_area = np.count_nonzero(total_mask > 1)

        overlap_ratio = total_overlap_area / total_sensing_area
        
        return overlap_ratio

    def _get_agent_explored_rates(self):
        if self.total_free_cells <= 0:
            return [0.0 for _ in range(self.n_agents)]
        rates = []
        for agent_id in range(self.n_agents):
            agent_map = self.env.agent_beliefs[agent_id]
            rate = np.count_nonzero(agent_map == FREE) / self.total_free_cells
            rates.append(float(rate))
        return rates

    def get_detected_robots_in_fov(self, robot, robot_locations, robot_headings):
        """Helper function to detect which robots are in the FOV of a given robot"""
        detected_robots = []
        robot_loc = get_coords_from_cell_position(robot_locations[robot.id], self.env.belief_info)

        for other_robot in self.robot_list:
            if other_robot.id == robot.id:
                continue

            other_loc = get_coords_from_cell_position(robot_locations[other_robot.id], self.env.belief_info)

            # Calculate distance
            distance = np.linalg.norm(other_loc - robot_loc)

            # Check if within sensor range
            if distance > self.sensor_range:
                continue

            # Calculate angle to the other robot
            delta = other_loc - robot_loc
            angle_to_robot = np.degrees(np.arctan2(delta[1], delta[0])) % 360

            # Calculate angle difference considering FOV
            angle_diff = (angle_to_robot - robot_headings[robot.id] + 180) % 360 - 180

            # Check if within FOV
            if np.abs(angle_diff) <= self.fov / 2:
                detected_robots.append(other_robot.id)

        return detected_robots

    def plot_individual_agent_views(self, step, robot_locations, robot_headings):
        """
        Create visualization showing each agent's individual map belief and frontier distribution
        to demonstrate they have different information due to decentralized learning.
        """
        plt.switch_backend('agg')

        # Layout: 3 rows - Global view, Individual maps, Frontier distributions
        n_agents = self.n_agents
        fig = plt.figure(figsize=(4 * n_agents, 12))

        color_list = ['r', 'b', 'g', 'y', 'c', 'm']
        color_name = ['Red', 'Blue', 'Green', 'Yellow', 'Cyan', 'Magenta']

        # Row 1: Global view for reference
        plt.subplot(3, n_agents, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.title('Global Environment', fontsize=12, fontweight='bold')
        plt.axis('off')

        # Draw all agents on global view
        for i, (robot, location, heading) in enumerate(zip(self.robot_list, robot_locations, robot_headings)):
            plot_id = robot.id % len(color_list)
            c = color_list[plot_id]
            plt.plot(location[0], location[1], f'{c}o', markersize=8, zorder=5)

        # Row 2: Each agent's individual map belief
        for robot in self.robot_list:
            plt.subplot(3, n_agents, n_agents + robot.id + 1)
            # Get agent's individual observation map (NOT the merged map)
            agent_map = self.individual_maps[robot.id]
            plt.imshow(agent_map, cmap='gray')
            plt.title(f'Agent {robot.id} Map Belief', fontsize=10, fontweight='bold')
            plt.axis('off')

            # Show agent's current location
            agent_location = robot_locations[robot.id]
            plot_id = robot.id % len(color_list)
            c = color_list[plot_id]
            plt.plot(agent_location[0], agent_location[1], f'{c}o', markersize=6, zorder=5)

            # Show agent's nodes and utilities
            try:
                if hasattr(robot.node_manager, 'nodes_dict') and robot.node_manager.nodes_dict:
                    node_coords = []
                    node_utilities = []
                    for node_item in robot.node_manager.nodes_dict:
                        node = node_item.data
                        coords_cell = get_cell_position_from_coords(np.array(node.coords), robot.map_info)
                        node_coords.append(coords_cell)
                        node_utilities.append(node.utility)

                    if node_coords:
                        node_coords = np.array(node_coords)
                        node_utilities = np.array(node_utilities)

                        # Show high utility nodes
                        high_utility_mask = node_utilities > 0
                        if np.any(high_utility_mask):
                            plt.scatter(node_coords[high_utility_mask, 0], node_coords[high_utility_mask, 1],
                                       c='orange', s=20, alpha=0.7, zorder=3, label='High Utility Nodes')
            except Exception as e:
                print(f"Warning: Could not visualize nodes for agent {robot.id}: {e}")

        # Row 3: Each agent's frontier distribution
        for robot in self.robot_list:
            plt.subplot(3, n_agents, 2 * n_agents + robot.id + 1)

            # Get agent's frontier distribution from their node manager
            try:
                # Update planning state to get latest frontier distribution
                robot.update_planning_state()

                if hasattr(robot, 'frontier_distribution') and robot.frontier_distribution is not None:
                    frontier_dist = robot.frontier_distribution
                    # Ensure frontier_dist is a 1D array
                    if frontier_dist.ndim > 1:
                        frontier_dist = frontier_dist.flatten()

                    angles = np.arange(0, 360, 360 / len(frontier_dist))

                    # Create polar plot for frontier distribution
                    ax = plt.subplot(3, n_agents, 2 * n_agents + robot.id + 1, projection='polar')
                    ax.plot(np.radians(angles), frontier_dist, color=color_list[robot.id % len(color_list)], linewidth=2)
                    ax.fill(np.radians(angles), frontier_dist, color=color_list[robot.id % len(color_list)], alpha=0.3)
                    ax.set_title(f'Agent {robot.id} Frontier Distribution', fontsize=10, fontweight='bold', pad=20)
                    ax.set_theta_zero_location('N')  # 0 degrees at top
                    ax.set_theta_direction(-1)  # Clockwise
                    max_val = max(frontier_dist) if max(frontier_dist) > 0 else 1
                    ax.set_ylim(0, max_val * 1.1)

                    # Add frontier count info
                    total_frontiers = np.sum(frontier_dist)
                    ax.text(0.02, 0.98, f'Total: {total_frontiers:.1f}', transform=ax.transAxes,
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    plt.text(0.5, 0.5, f'No frontier data\nfor Agent {robot.id}',
                            ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
                    plt.axis('off')
            except Exception as e:
                plt.text(0.5, 0.5, f'Error visualizing\nAgent {robot.id}\nFrontier Distribution\n{str(e)[:30]}...',
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=8)
                plt.axis('off')

        # Add statistics for comparison
        stats_text = "Agent Comparison Stats:\n"
        for robot in self.robot_list:
            try:
                if hasattr(robot, 'frontier_distribution') and robot.frontier_distribution is not None:
                    total_frontiers = np.sum(robot.frontier_distribution)
                    max_frontier = np.max(robot.frontier_distribution)
                    stats_text += f"Agent {robot.id}: Total={total_frontiers:.1f}, Max={max_frontier:.1f}\n"
                else:
                    stats_text += f"Agent {robot.id}: No frontier data\n"
            except:
                stats_text += f"Agent {robot.id}: Error getting data\n"

        # Add text box with statistics
        fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Overall title
        plt.suptitle(f'Step {step}: Individual Agent Views (Decentralized Learning)\n'
                    f'Each agent has independent map beliefs and frontier distributions',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save the frame
        # if self.save_image:
        #     plt.savefig(gifs_path + f'/individual_views_{self.global_step}_{step}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _draw_training_style_robot_overlay(self, ax, location, heading, color, sensing_range, draw_fov=True, linewidth=1.2):
        location = np.asarray(location)
        ax.plot(location[0], location[1], marker='o', color=color, markersize=4.5, zorder=6)
        dx, dy = self.heading_to_vector(heading, length=sensing_range)
        arrow = FancyArrowPatch(
            (location[0], location[1]),
            (location[0] + dx / 1.25, location[1] + dy / 1.25),
            mutation_scale=10,
            linewidth=linewidth,
            color=color,
            arrowstyle='-|>',
            zorder=7,
        )
        ax.add_artist(arrow)
        if draw_fov:
            cone = Wedge(
                center=(location[0], location[1]),
                r=sensing_range,
                theta1=heading - self.fov / 2,
                theta2=heading + self.fov / 2,
                color=color,
                alpha=0.22,
                zorder=5,
            )
            ax.add_artist(cone)

    @staticmethod
    def _get_test_node_plot_data(node_manager):
        node_coords = []
        node_utility = []
        for node in node_manager.nodes_dict.__iter__():
            node_coords.append(node.data.coords)
            node_utility.append(node.data.utility)
        if not node_coords:
            return None, None
        return np.asarray(node_coords).reshape(-1, 2), np.asarray(node_utility)

    def _draw_frontiers(self, ax, map_info, size=2):
        frontiers = get_frontier_in_map(map_info)
        if len(frontiers) == 0:
            return
        frontier_cells = get_cell_position_from_coords(np.array(list(frontiers)), map_info)
        if len(frontiers) == 1:
            frontier_cells = frontier_cells.reshape(1, 2)
        ax.scatter(frontier_cells[:, 0], frontier_cells[:, 1], s=size, c='r', zorder=4)

    def _draw_test_nodes(self, ax, node_coords, node_utility, map_info, color):
        if node_coords is None or len(node_coords) == 0:
            return
        nodes = get_cell_position_from_coords(node_coords, map_info)
        if nodes.ndim == 1:
            nodes = nodes.reshape(1, 2)
        ax.scatter(nodes[:, 0], nodes[:, 1], c=color, s=8, zorder=3, alpha=0.65)
        utility_mask = node_utility > 0
        if np.any(utility_mask):
            ax.scatter(nodes[utility_mask, 0], nodes[utility_mask, 1], c='orange', s=20, zorder=4, alpha=0.8)

    def plot_local_env_sim(self, step, robot_locations, robot_headings):
        plt.switch_backend('agg')

        n_cols = max(4, self.n_agents)
        fig = plt.figure(figsize=(15, 7.5), constrained_layout=True)
        gs = fig.add_gridspec(2, n_cols, height_ratios=[1.0, 1.05])
        color_list = ['tab:red', 'tab:blue', 'tab:green', 'goldenrod', 'tab:purple', 'tab:brown']
        color_name = ['Red', 'Blue', 'Green', 'Yellow', 'Purple', 'Brown']
        sensing_range = self.sensor_range / CELL_SIZE
        plot_robot_locations = np.asarray(robot_locations)
        total_free = np.sum(self.env.ground_truth == FREE)
        agent_map_infos = [self.env.get_agent_map_info(robot.id) for robot in self.robot_list]
        agent_node_data = [
            self._get_test_node_plot_data(robot.node_manager)
            for robot in self.robot_list
        ]

        merged_ax = fig.add_subplot(gs[0, 0])
        merged_ax.imshow(self.env.robot_belief, cmap='gray', interpolation='nearest', vmin=OCCUPIED, vmax=FREE)
        merged_ax.axis('off')
        merged_ax.set_title('Merged Team Belief', fontsize=10, fontweight='bold')
        xlim = merged_ax.get_xlim()
        ylim = merged_ax.get_ylim()
        merged_ax.set_xlim(xlim[0], xlim[1])
        merged_ax.set_ylim(ylim[0], ylim[1])
        self._draw_frontiers(merged_ax, self.env.belief_info, size=2)

        for robot, location, heading in zip(self.robot_list, plot_robot_locations, robot_headings):
            c = color_list[robot.id % len(color_list)]
            self._draw_training_style_robot_overlay(merged_ax, location, heading, c, sensing_range)
            if self.broken_agents[robot.id]:
                self._plot_breakdown_label(location)

        fov_ax = fig.add_subplot(gs[0, 1])
        fov_ax.imshow(self.env.robot_belief, cmap='gray', interpolation='nearest', vmin=OCCUPIED, vmax=FREE)
        fov_ax.axis('off')
        fov_ax.set_xlim(xlim[0], xlim[1])
        fov_ax.set_ylim(ylim[0], ylim[1])
        fov_ax.set_title('Team Motion + FoV', fontsize=10, fontweight='bold')
        self._draw_frontiers(fov_ax, self.env.belief_info, size=2)

        for robot, location, heading in zip(self.robot_list, plot_robot_locations, robot_headings):
            c = color_list[robot.id % len(color_list)]
            robot_location = get_coords_from_cell_position(location, self.env.belief_info)
            trajectory_x = robot.trajectory_x.copy()
            trajectory_y = robot.trajectory_y.copy()
            trajectory_x.append(robot_location[0])
            trajectory_y.append(robot_location[1])
            fov_ax.plot(
                (np.array(trajectory_x) - self.env.belief_info.map_origin_x) / CELL_SIZE,
                (np.array(trajectory_y) - self.env.belief_info.map_origin_y) / CELL_SIZE,
                c,
                linewidth=1.2,
                zorder=1,
            )
            self._draw_training_style_robot_overlay(fov_ax, location, heading, c, sensing_range)
            if self.broken_agents[robot.id]:
                self._plot_breakdown_label(location)

        gt_ax = fig.add_subplot(gs[0, 2])
        gt_ax.imshow(self.env.ground_truth, cmap='gray', interpolation='nearest', vmin=OCCUPIED, vmax=FREE)
        gt_ax.set_xlim(xlim[0], xlim[1])
        gt_ax.set_ylim(ylim[0], ylim[1])
        gt_ax.axis('off')
        gt_ax.set_title('Ground Truth', fontsize=10, fontweight='bold')
        for robot, location, heading in zip(self.robot_list, plot_robot_locations, robot_headings):
            c = color_list[robot.id % len(color_list)]
            self._draw_training_style_robot_overlay(gt_ax, location, heading, c, sensing_range)
            if self.broken_agents[robot.id]:
                self._plot_breakdown_label(location)

        summary_ax = fig.add_subplot(gs[0, 3])
        summary_ax.axis('off')
        summary_ax.set_xlim(0, 1)
        summary_ax.set_ylim(0, 1)
        summary_ax.add_patch(Rectangle((0.01, 0.01), 0.98, 0.98, facecolor='#f7f5ef', edgecolor='#c8c2b5', linewidth=1.4))
        summary_ax.text(0.08, 0.96, 'Legend & Stats', fontsize=11.5, fontweight='bold', color='#2b2b2b', va='top')

        legend_rows = [
            [('Occupied', '#101010'), ('Unknown', '#7f7f7f')],
            [('Free', '#f5f5f5'), ('Utility Node', 'orange')],
            [('Frontier', 'red'), ('Broken', 'black')],
        ]
        for row_y, row_items in zip([0.84, 0.77, 0.70], legend_rows):
            for col_x, item in zip([0.08, 0.53], row_items):
                label, facecolor = item
                summary_ax.add_patch(Rectangle((col_x, row_y - 0.020), 0.055, 0.034, facecolor=facecolor, edgecolor='#333333', linewidth=0.8))
                summary_ax.text(col_x + 0.09, row_y - 0.003, label, fontsize=9.0, fontweight='bold', color='#2b2b2b', va='center')

        agent_explored_rates = self._get_agent_explored_rates()
        avg_agent_explored_rate = float(np.mean(agent_explored_rates))
        active_agent_ids = self._active_agent_ids_for_merged_map()
        broken_ids = [int(agent_id) for agent_id in np.where(self.broken_agents)[0]]

        summary_ax.plot([0.08, 0.92], [0.61, 0.61], color='#d8d2c7', linewidth=1.0)
        summary_ax.text(0.08, 0.58, 'Episode', fontsize=10, fontweight='bold', color='#2b2b2b', va='top')
        rows = [
            ('Merged explored', f'{self.env.explored_rate:.1%}'),
            ('Avg agent explored', f'{avg_agent_explored_rate:.1%}'),
            ('Max travel dist', f'{max([robot.travel_dist for robot in self.robot_list]):.1f}'),
            ('Merged uses agents', ','.join(map(str, active_agent_ids)) if active_agent_ids else 'none'),
            ('Broken agents', ','.join(map(str, broken_ids)) if broken_ids else 'none'),
        ]
        row_y = 0.50
        for label_text, value_text in rows:
            summary_ax.text(0.08, row_y, label_text, fontsize=8.7, fontweight='bold', color='#4a4a4a', va='center')
            summary_ax.text(0.86, row_y, value_text, fontsize=8.9, fontweight='bold', color='#2b2b2b', va='center', ha='right')
            row_y -= 0.07

        summary_ax.plot([0.08, 0.92], [0.14, 0.14], color='#d8d2c7', linewidth=1.0)
        summary_ax.text(0.08, 0.10, 'Breakdown map removal', fontsize=8.8, fontweight='bold', color='#2b2b2b', va='center')
        removal_text = 'merged map excludes broken beliefs' if broken_ids else 'no broken agents yet'
        summary_ax.text(0.92, 0.04, removal_text, fontsize=8.0, fontweight='bold', color='#2b2b2b', va='center', ha='right')

        for robot, agent_explored_rate, agent_map_info, plot_node_data in zip(
            self.robot_list,
            agent_explored_rates,
            agent_map_infos,
            agent_node_data,
        ):
            agent_ax = fig.add_subplot(gs[1, robot.id])
            color_idx = robot.id % len(color_list)
            c = color_list[color_idx]
            agent_map = agent_map_info.map
            agent_ax.imshow(agent_map, cmap='gray', interpolation='nearest', vmin=OCCUPIED, vmax=FREE)
            agent_ax.axis('off')
            agent_ax.set_xlim(xlim[0], xlim[1])
            agent_ax.set_ylim(ylim[0], ylim[1])

            robot_location = get_coords_from_cell_position(plot_robot_locations[robot.id], self.env.belief_info)
            trajectory_x = robot.trajectory_x.copy()
            trajectory_y = robot.trajectory_y.copy()
            trajectory_x.append(robot_location[0])
            trajectory_y.append(robot_location[1])
            agent_ax.plot(
                (np.array(trajectory_x) - self.env.belief_info.map_origin_x) / CELL_SIZE,
                (np.array(trajectory_y) - self.env.belief_info.map_origin_y) / CELL_SIZE,
                c,
                linewidth=1.5,
                alpha=0.9,
                zorder=2,
            )

            plot_node_coords, plot_node_utility = plot_node_data
            self._draw_test_nodes(agent_ax, plot_node_coords, plot_node_utility, agent_map_info, c)
            self._draw_frontiers(agent_ax, agent_map_info, size=2)

            location = plot_robot_locations[robot.id]
            heading = robot_headings[robot.id]
            self._draw_training_style_robot_overlay(agent_ax, location, heading, c, sensing_range)
            if self.broken_agents[robot.id]:
                self._plot_breakdown_label(location)

            num_nodes = 0 if plot_node_coords is None else len(plot_node_coords)
            title_suffix = '  BROKEN' if self.broken_agents[robot.id] else ''
            agent_ax.set_title(
                f'{color_name[color_idx]} Belief{title_suffix}\nExplored: {agent_explored_rate:.1%}  Nodes: {num_nodes}',
                fontsize=9,
                fontweight='bold',
                color=c,
            )

            initial_budget = max(float(self.initial_budgets[robot.id]), 1.0)
            remaining = float(max(self.remaining_budgets[robot.id], 0.0))
            fraction_remaining = remaining / initial_budget
            bar_color = '#2ecc71' if fraction_remaining > 0.5 else '#f39c12' if fraction_remaining > 0.25 else '#e74c3c'
            agent_ax.add_patch(Rectangle((0.0, 0.955), 1.0, 0.04, transform=agent_ax.transAxes, color='#d0d0d0', clip_on=True, zorder=15))
            agent_ax.add_patch(Rectangle((0.0, 0.955), fraction_remaining, 0.04, transform=agent_ax.transAxes, color=bar_color, clip_on=True, zorder=16))
            mode_text = ' | broken' if self.broken_agents[robot.id] else ' | return' if self.returning_agents[robot.id] else ''
            agent_ax.text(
                0.5,
                0.975,
                f'{remaining:.1f}/{initial_budget:.1f} m{mode_text}',
                transform=agent_ax.transAxes,
                fontsize=6.5,
                ha='center',
                va='center',
                fontweight='bold',
                color='#1a1a1a',
                clip_on=True,
                zorder=17,
            )

        for empty_col in range(self.n_agents, n_cols):
            empty_ax = fig.add_subplot(gs[1, empty_col])
            empty_ax.axis('off')

        robot_headings_str = [f"{color_name[robot.id % len(color_name)]} {robot.heading:.0f} deg" for robot in self.robot_list]
        fig.suptitle(
            'Experiment ID: {}\nRobot headings: {}'.format(LOAD_FOLDER_NAME, ', '.join(robot_headings_str)),
            fontweight='bold',
            fontsize=11,
        )
        frame = '{}/{}_{}_{}_{}_{}_samples.png'.format(gifs_path, self.global_step, step, self.n_agents, self.fov, self.sensor_range)
        plt.savefig(frame, dpi=150, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        gc.collect()
        self.env.frame_files.append(frame)

    def correct_heading(self, heading):
        heading = abs(((heading + 90) % 360) - 360)
        return heading

if __name__ == '__main__':
    import torch
    from utils.model import PolicyNet

    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, use_trajectory=USE_TRAJECTORY)
    if LOAD_MODEL:
        checkpoint = torch.load(load_path, map_location='cpu')
        load_compatible_state_dict(policy_net, checkpoint['policy_model'], 'policy')
        print('Policy loaded!')
    worker = TestWorker(0, policy_net, 188, 4, 120, 10, 9.0, 'cpu', True)
    worker.run_episode()
