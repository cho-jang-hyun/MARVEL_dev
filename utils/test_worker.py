import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrowPatch
from skimage.draw import polygon as sk_polygon
from skimage.morphology import label
from collections import deque
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
            selected_locations = [robot.location.copy() for robot in self.robot_list]
            dist_list = [0.0 for _ in range(self.n_agents)]
            next_heading_index_list = [self._get_heading_index_towards(robot, robot.location) for robot in self.robot_list]
            observations = {}
            active_explorer_ids = []

            for robot in self.robot_list:
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

            if len(active_explorer_ids) == 0 and np.all(self.returning_agents):
                if not SIMULATE_RETURN_TO_BASE or self._all_agents_at_base():
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

            if len(active_explorer_ids) == 0 and np.all(self.returning_agents):
                if not SIMULATE_RETURN_TO_BASE or self._all_agents_at_base():
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

            if len(active_explorer_ids) == 0 and np.all(self.returning_agents):
                if not SIMULATE_RETURN_TO_BASE or self._all_agents_at_base():
                    break

            robot_locations_sim = []
            robot_headings_sim = []
            all_robots_heading_list = []
            for robot, next_location, next_heading_index in zip(self.robot_list, selected_locations, next_heading_index_list):
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
                    self.env.update_robot_belief(
                        q,
                        robot_locations_sim[q][l],
                        robot_headings_sim[q][l],
                        refresh_merged=False,
                    )
                    if self.save_image:
                        self.individual_maps[q] = self.env.get_agent_map_info(q).map.copy()
                    robot_location_sim_step.append(robot_locations_sim[q][l])
                    robot_heading_sim_step.append(robot_headings_sim[q][l])
                self.env.refresh_merged_belief()

                if self.save_image:
                    num_frame = i * self.sim_steps + l
                    self.plot_local_env_sim(num_frame, robot_location_sim_step, robot_heading_sim_step)
                    if num_frame % 5 == 0:
                        self.plot_individual_agent_views(num_frame, robot_location_sim_step, robot_heading_sim_step)

            previous_locations = [robot.location.copy() for robot in self.robot_list]
            for robot, next_location in zip(self.robot_list, selected_locations):
                self.env.final_sim_step(next_location, robot.id)
                traveled_distance = float(np.linalg.norm(previous_locations[robot.id] - next_location))
                if traveled_distance > 0.0:
                    self.remaining_budgets[robot.id] -= traveled_distance
                self._append_trajectory_step(robot, next_location)

            for robot in self.robot_list:
                robot.update_graph(self.env.get_agent_map_info(robot.id), self.env.robot_locations[robot.id].copy())
            for robot in self.robot_list:
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
            if np.all(self.returning_agents):
                if not SIMULATE_RETURN_TO_BASE or self._all_agents_at_base():
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

    def plot_local_env_sim(self, step, robot_locations, robot_headings):
        plt.switch_backend('agg')

        # Layout: Top row - merged map + FOV view, Bottom row - each agent's individual map
        n_cols = max(2, self.n_agents)
        fig = plt.figure(figsize=(3 * n_cols, 9))

        color_list = ['r', 'b', 'g', 'y', 'c', 'm']
        color_name = ['Red', 'Blue', 'Green', 'Yellow', 'Cyan', 'Magenta']
        sensing_range = self.sensor_range / CELL_SIZE

        # Detect robots in FOV for each robot
        fov_detections = {}
        for robot in self.robot_list:
            fov_detections[robot.id] = self.get_detected_robots_in_fov(robot, robot_locations, robot_headings)

        # Top row - Panel 1: Global merged belief map
        plt.subplot(3, n_cols, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.title('Merged Map (Ground Truth)', fontsize=10, fontweight='bold')

        # Draw trajectories and arrows on merged map
        for i, (robot, location, heading) in enumerate(zip(self.robot_list, robot_locations, robot_headings)):
            plot_id = robot.id % len(color_list)
            c = color_list[plot_id]
            robot_location = get_coords_from_cell_position(location, self.env.belief_info)
            trajectory_x = robot.trajectory_x.copy()
            trajectory_y = robot.trajectory_y.copy()
            trajectory_x.append(robot_location[0])
            trajectory_y.append(robot_location[1])
            plt.plot((np.array(trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
                     (np.array(trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size, c,
                     linewidth=1.5, alpha=0.7, zorder=1)

            dx, dy = self.heading_to_vector(heading, length=sensing_range)
            arrow = FancyArrowPatch((location[0], location[1]), (location[0] + dx/1.25, location[1] + dy/1.25),
                                    mutation_scale=10, color=c, arrowstyle='-|>')
            plt.gca().add_artist(arrow)

        global_frontiers = get_frontier_in_map(self.env.belief_info)
        if len(global_frontiers) != 0:
            frontiers_cell = get_cell_position_from_coords(np.array(list(global_frontiers)), self.env.belief_info)
            if len(global_frontiers) == 1:
                frontiers_cell = frontiers_cell.reshape(1,2)
            plt.scatter(frontiers_cell[:, 0], frontiers_cell[:, 1], s=1, c='r')

        # Top row - Panel 2: FOV & Detections view
        plt.subplot(3, n_cols, 2)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.title('FOV & Detections', fontsize=10, fontweight='bold')

        for i, (robot, location, heading) in enumerate(zip(self.robot_list, robot_locations, robot_headings)):
            plot_id = robot.id % len(color_list)
            c = color_list[plot_id]
            dx, dy = self.heading_to_vector(heading, length=sensing_range)
            arrow = FancyArrowPatch((location[0], location[1]), (location[0] + dx/1.25, location[1] + dy/1.25),
                                    mutation_scale=10, color=c, arrowstyle='-|>')
            plt.gca().add_artist(arrow)
            cone = Wedge(center=(location[0], location[1]), r=self.sensor_range / CELL_SIZE,
                        theta1=(heading-self.fov/2), theta2=(heading+self.fov/2), color=c, alpha=0.3, zorder=10)
            plt.gca().add_artist(cone)

        # Draw detection connections
        for detector_id, detected_list in fov_detections.items():
            for detected_id in detected_list:
                detector_loc = robot_locations[detector_id]
                detected_loc = robot_locations[detected_id]
                plt.plot([detector_loc[0], detected_loc[0]], [detector_loc[1], detected_loc[1]],
                         'w--', linewidth=1.5, alpha=0.6, zorder=11)

        if len(global_frontiers) != 0:
            plt.scatter(frontiers_cell[:, 0], frontiers_cell[:, 1], s=3, c='r')

        # Middle row: Each agent's INDIVIDUAL partially observed map (full view)
        for robot in self.robot_list:
            plt.subplot(3, n_cols, n_cols + robot.id + 1)
            plot_id = robot.id % len(color_list)
            c = color_list[plot_id]

            # Get this agent's individual observation map (NOT the merged map)
            agent_map = self.individual_maps[robot.id].copy()

            plt.imshow(agent_map, cmap='gray')
            plt.axis('off')

            # Draw this agent's trajectory
            robot_location = get_coords_from_cell_position(robot_locations[robot.id], self.env.belief_info)
            trajectory_x = robot.trajectory_x.copy()
            trajectory_y = robot.trajectory_y.copy()
            trajectory_x.append(robot_location[0])
            trajectory_y.append(robot_location[1])
            plt.plot((np.array(trajectory_x) - self.env.belief_info.map_origin_x) / CELL_SIZE,
                     (np.array(trajectory_y) - self.env.belief_info.map_origin_y) / CELL_SIZE, c,
                     linewidth=2, alpha=0.9, zorder=2)

            # Draw current position and heading
            location = robot_locations[robot.id]
            heading = robot_headings[robot.id]
            dx, dy = self.heading_to_vector(heading, length=sensing_range)
            arrow = FancyArrowPatch((location[0], location[1]), (location[0] + dx/1.25, location[1] + dy/1.25),
                                    mutation_scale=10, color=c, arrowstyle='-|>', linewidth=2)
            plt.gca().add_artist(arrow)

            # Draw FOV cone
            cone = Wedge(center=(location[0], location[1]), r=self.sensor_range / CELL_SIZE,
                        theta1=(heading - self.fov/2), theta2=(heading + self.fov/2),
                        color=c, alpha=0.3, zorder=10)
            plt.gca().add_artist(cone)

            # Draw frontiers from this agent's individual map
            agent_map_info = MapInfo(agent_map, self.env.belief_info.map_origin_x,
                                     self.env.belief_info.map_origin_y, CELL_SIZE)
            agent_frontiers_set = get_frontier_in_map(agent_map_info)
            if len(agent_frontiers_set) > 0:
                agent_frontiers = get_cell_position_from_coords(
                    np.array(list(agent_frontiers_set)), agent_map_info)
                if len(agent_frontiers_set) == 1:
                    agent_frontiers = agent_frontiers.reshape(1, 2)
                plt.scatter(agent_frontiers[:, 0], agent_frontiers[:, 1], s=3, c='r', zorder=8)

            # Calculate explored percentage for this agent
            total_free = np.sum(self.env.ground_truth == FREE)
            agent_explored = np.sum(agent_map == FREE)
            agent_explored_rate = agent_explored / total_free if total_free > 0 else 0

            plt.title(f'{color_name[plot_id]} Agent Map\nExplored: {agent_explored_rate:.1%}',
                     fontsize=9, fontweight='bold', color=c)

        # Bottom row: Individual agent local views (zoomed)
        local_map_size = int(UPDATING_MAP_SIZE / CELL_SIZE)

        for robot in self.robot_list:
            plt.subplot(3, n_cols, 2 * n_cols + robot.id + 1)
            plot_id = robot.id % len(color_list)
            c = color_list[plot_id]

            # Use agent's individual observation map for local view
            agent_map = self.individual_maps[robot.id]
            center_cell = robot_locations[robot.id]
            half_size = local_map_size // 2

            row_start = max(0, int(center_cell[1] - half_size))
            row_end = min(agent_map.shape[0], int(center_cell[1] + half_size))
            col_start = max(0, int(center_cell[0] - half_size))
            col_end = min(agent_map.shape[1], int(center_cell[0] + half_size))

            local_map = agent_map[row_start:row_end, col_start:col_end]
            plt.imshow(local_map, cmap='gray')
            plt.axis('off')

            # Calculate robot position in local map coordinates
            robot_local_x = center_cell[0] - col_start
            robot_local_y = center_cell[1] - row_start

            # Draw robot position and heading
            dx, dy = self.heading_to_vector(robot_headings[robot.id], length=sensing_range)
            arrow = FancyArrowPatch(
                (robot_local_x, robot_local_y),
                (robot_local_x + dx/1.25, robot_local_y + dy/1.25),
                mutation_scale=10, color=c, arrowstyle='-|>', linewidth=2
            )
            plt.gca().add_artist(arrow)

            # Draw FOV cone
            cone = Wedge(
                center=(robot_local_x, robot_local_y),
                r=self.sensor_range / CELL_SIZE,
                theta1=(robot_headings[robot.id] - self.fov/2),
                theta2=(robot_headings[robot.id] + self.fov/2),
                color=c, alpha=0.3, zorder=10
            )
            plt.gca().add_artist(cone)

            # Draw other robots if they are in this local view
            for other_robot in self.robot_list:
                if other_robot.id == robot.id:
                    continue

                other_location = robot_locations[other_robot.id]
                other_local_x = other_location[0] - col_start
                other_local_y = other_location[1] - row_start

                if 0 <= other_local_x < local_map.shape[1] and 0 <= other_local_y < local_map.shape[0]:
                    other_plot_id = other_robot.id % len(color_list)
                    other_c = color_list[other_plot_id]
                    is_detected = other_robot.id in fov_detections.get(robot.id, [])

                    if is_detected:
                        plt.plot(other_local_x, other_local_y, 'o', color=other_c, markersize=10,
                                markeredgewidth=3, markeredgecolor='yellow', zorder=15)
                        plt.plot([robot_local_x, other_local_x], [robot_local_y, other_local_y],
                                'y--', linewidth=2, alpha=0.8, zorder=12)
                    else:
                        plt.plot(other_local_x, other_local_y, 'o', color=other_c, markersize=6, alpha=0.5, zorder=5)

            # Draw local frontiers from agent's individual map
            agent_map_info = MapInfo(agent_map, self.env.belief_info.map_origin_x,
                                     self.env.belief_info.map_origin_y, CELL_SIZE)
            agent_frontiers_set = get_frontier_in_map(agent_map_info)
            if len(agent_frontiers_set) > 0:
                local_frontiers = []
                for frontier_coords in agent_frontiers_set:
                    frontier_cell = get_cell_position_from_coords(np.array(frontier_coords), agent_map_info)
                    frontier_local_x = frontier_cell[0] - col_start
                    frontier_local_y = frontier_cell[1] - row_start
                    if 0 <= frontier_local_x < local_map.shape[1] and 0 <= frontier_local_y < local_map.shape[0]:
                        local_frontiers.append([frontier_local_x, frontier_local_y])

                if local_frontiers:
                    local_frontiers = np.array(local_frontiers)
                    plt.scatter(local_frontiers[:, 0], local_frontiers[:, 1], s=2, c='r', zorder=8)

            detected_names = [color_name[did % len(color_list)] for did in fov_detections.get(robot.id, [])]
            detected_text = f"Sees: {', '.join(detected_names)}" if detected_names else "No detections"
            plt.title(f'{color_name[plot_id]} Local View\n{detected_text}', fontsize=9, fontweight='bold', color=c)

        # Build detection summary
        detection_summary = []
        for robot_id, detected_list in fov_detections.items():
            if len(detected_list) > 0:
                detected_colors = [color_name[did % len(color_list)] for did in detected_list]
                detection_summary.append(f"{color_name[robot_id % len(color_list)]}: {', '.join(detected_colors)}")

        detection_text = ' | '.join(detection_summary) if detection_summary else 'No detections'

        robot_headings_text = [f"{color_name[robot.id % len(color_list)]}: {robot.heading:.0f}°" for robot in self.robot_list]
        plt.suptitle('Explored: {:.4g}  Distance: {:.4g}\nHeadings: {}\nFOV Detections: {}'.format(
            self.env.explored_rate,
            max([robot.travel_dist for robot in self.robot_list]),
            ', '.join(robot_headings_text),
            detection_text
        ), fontweight='bold', fontsize=10, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('{}/{}_{}_{}_{}_{}_samples.png'.format(gifs_path, self.global_step, step, self.n_agents, self.fov, self.sensor_range), dpi=150)
        plt.close()
        frame = '{}/{}_{}_{}_{}_{}_samples.png'.format(gifs_path, self.global_step, step, self.n_agents, self.fov, self.sensor_range)
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
