"""
A multi-agent worker class for coordinating multi-robots exploration in an indoor environment.

This class manages a group of agents performing collaborative exploration, handling
their movement, observation, reward calculation, and simulation steps. It supports
features like collision avoidance, trajectory planning, and performance tracking.

Key functionalities:
- Initializes multiple agents with a shared policy network
- Runs exploration episodes with collision resolution
- Tracks agent locations, headings, and exploration progress
- Generates visualizations of the exploration process
- Calculates rewards and saves episode data

Attributes:
    meta_agent_id (int): Identifier for the meta-agent group
    global_step (int): Current global simulation step
    env (Env): Environment simulation instance
    robot_list (List[Agent]): List of agents in the exploration team
    episode_buffer (List): Buffer for storing episode data
    perf_metrics (dict): Performance metrics for the episode
    trajectory_buffer (dict): Stores recent trajectory history for each agent
"""
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.patches import Wedge, FancyArrowPatch, Rectangle
from collections import deque
import torch

from utils.env import Env
from utils.agent import Agent
from utils.utils import *
from utils.node_manager import NodeManager
from utils.ground_truth_node_manager import GroundTruthNodeManager
from utils.merged_critic_manager import MergedBeliefCriticManager
from utils.model import PolicyNet
from utils.motion_model import compute_allowable_heading  

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

class MultiAgentWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False, curriculum_success_rate=0.0):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.fov = FOV
        self.sensor_range = SENSOR_RANGE
        self.sim_steps = NUM_SIM_STEPS

        self.env = Env(global_step, self.fov, self.sensor_range, plot=self.save_image)
        self.n_agents = N_AGENTS
        self.use_merged_critic = TRAIN_ALGO == 4
        self.merged_map_manager = MergedBeliefCriticManager(
            self.fov,
            self.sensor_range,
            device=self.device,
            plot=self.save_image,
        )
        self.merged_critic_manager = self.merged_map_manager if self.use_merged_critic else None

        # Create independent node managers for each agent to ensure decentralized learning
        self.robot_list = []
        for i in range(self.n_agents):
            # Each agent gets its own independent node_manager and ground_truth_node_manager
            individual_node_manager = NodeManager(self.fov, self.sensor_range, plot=self.save_image)
            individual_ground_truth_node_manager = GroundTruthNodeManager(individual_node_manager, self.env.ground_truth_info, self.sensor_range,
                                                                          device=self.device, plot=self.save_image)

            agent = Agent(i, policy_net, self.fov, self.env.angles[i], self.sensor_range,
                         individual_node_manager, individual_ground_truth_node_manager,
                         self.device, self.save_image)
            self.robot_list.append(agent)

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(NUM_EPISODE_BUFFER):
            self.episode_buffer.append([])

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
        self.initial_budgets = self._sample_initial_budgets(curriculum_success_rate)
        self.remaining_budgets = self.initial_budgets.copy()
        self.returning_agents = np.zeros(self.n_agents, dtype=bool)
        self.replay_closed = np.zeros(self.n_agents, dtype=bool)
        self.low_utility_streaks = np.zeros(self.n_agents, dtype=int)
        self.individual_objective_completed = np.zeros(self.n_agents, dtype=bool)
        self.merged_objective_completed = False
        self.merged_completion_travel_dist = None
        self.merged_total_utility = np.inf

    def _get_collision_free_candidate(self, robot, occupied_locations):
        occupied_keys = {
            (float(location[0]), float(location[1]))
            for location in np.asarray(occupied_locations).reshape(-1, 2)
        }

        for _, _, candidate_location, candidate_heading_index in robot.current_waypoint_candidates:
            candidate_key = (float(candidate_location[0]), float(candidate_location[1]))
            if candidate_key in occupied_keys:
                continue
            return candidate_location.copy(), candidate_heading_index

        return None, None

    def _sample_initial_budgets(self, curriculum_success_rate=0.0):
        success_rate = float(np.clip(curriculum_success_rate, 0.0, 1.0))
        target_budget = BUDGET_END + (BUDGET_START - BUDGET_END) * (1.0 - success_rate)

        if np.random.random() < BUDGET_CURRICULUM_UNIFORM_P:
            sampled_budgets = np.random.uniform(BUDGET_END, BUDGET_START, size=self.n_agents)
        else:
            noise = np.random.uniform(-BUDGET_CURRICULUM_NOISE, BUDGET_CURRICULUM_NOISE, size=self.n_agents)
            sampled_budgets = target_budget + noise

        return np.clip(sampled_budgets, BUDGET_END, BUDGET_START).astype(float)

    def _set_agent_budget_context(self, robot):
        robot.set_budget_context(
            self.initial_budgets[robot.id],
            self.remaining_budgets[robot.id],
            self.base_locations[robot.id],
            return_mode=self.returning_agents[robot.id],
        )

    def _update_merged_graph(self):
        self.merged_map_manager.update_graph(self.env.belief_info, self.env.robot_locations)

    def _get_max_travel_dist(self):
        return max((robot.travel_dist for robot in self.robot_list), default=0.0)

    def _update_objective_state(self):
        self.merged_total_utility = float(self.merged_map_manager.get_total_utility())
        merged_completed_this_step = (
            not self.merged_objective_completed
            and self.env.explored_rate > SUCCESS_THRESHOLD
        )
        if merged_completed_this_step:
            self.merged_objective_completed = True
            self.merged_completion_travel_dist = self._get_max_travel_dist()
        return merged_completed_this_step

    def _local_objective_completed(self, robot):
        total_free = self.env.total_free_cells
        if total_free <= 0:
            return False
        agent_map = self.env.get_agent_map_info(robot.id).map
        agent_explored_rate = np.sum(agent_map == FREE) / total_free
        return bool(agent_explored_rate > SUCCESS_THRESHOLD)

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

    def _get_heading_index_towards(self, robot, next_location):
        next_location = np.asarray(next_location, dtype=float)
        if np.allclose(next_location, robot.location):
            return int((robot.heading % 360) / 360 * NUM_ANGLES_BIN) % NUM_ANGLES_BIN
        angle = np.degrees(np.arctan2(
            next_location[1] - robot.location[1],
            next_location[0] - robot.location[0],
        ) % (2 * np.pi))
        return int(angle / 360 * NUM_ANGLES_BIN) % NUM_ANGLES_BIN

    def _get_return_path(self, robot, start_location=None):
        self._set_agent_budget_context(robot)
        return robot.get_path_to_base(start_location=start_location)

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

    def _episode_success(self, mission_failure):
        return bool(self.merged_objective_completed)

    def _get_joint_next_index_lists(self):
        local_next_indices = [robot.current_index for robot in self.robot_list]
        critic_next_indices = None
        if self.use_merged_critic:
            critic_next_indices = [self.merged_critic_manager.get_node_index(robot.location) for robot in self.robot_list]
        return local_next_indices, critic_next_indices

    def _close_agent_replay(self, robot, team_node_managers):
        if self.replay_closed[robot.id] or len(robot.episode_buffer[0]) == 0:
            self.replay_closed[robot.id] = True
            return

        self._set_agent_budget_context(robot)
        observation = robot.get_observation(
            robot_locations=self.env.robot_locations,
            trajectory_buffer=self.trajectory_buffer
        )
        local_next_indices, critic_next_indices = self._get_joint_next_index_lists()
        joint_next_index_list = local_next_indices
        if self.use_merged_critic and USE_COMMUNICATION:
            joint_next_index_list = critic_next_indices
        robot.save_next_observations(observation, joint_next_index_list)

        if self.use_merged_critic:
            critic_observation = self.merged_critic_manager.get_critic_observation(
                robot.location,
                robot.id,
                self.env.robot_locations,
                team_node_managers,
                self.env.get_agent_map_info(robot.id),
                local_observation=observation,
                local_node_coords=robot.node_coords,
                base_location=self.base_locations[robot.id],
            )
            robot.save_next_critic_observations(critic_observation)
        else:
            ground_truth_observation = robot.ground_truth_node_manager.get_ground_truth_observation(robot.location, base_location=self.base_locations[robot.id])
            robot.save_next_ground_truth_observations(ground_truth_observation)

        self.replay_closed[robot.id] = True

    def _finalize_open_agent_replays(self, team_node_managers):
        for robot in self.robot_list:
            if self.replay_closed[robot.id] or len(robot.episode_buffer[0]) == 0:
                continue
            self._close_agent_replay(robot, team_node_managers)

    def _has_feasible_exploration_action(self, robot, observation):
        current_edge = observation[4][0, :, 0].detach().cpu().numpy()
        edge_padding = observation[5][0, 0].detach().cpu().numpy()

        valid_slots = np.where(edge_padding == 0)[0]
        for slot in valid_slots:
            if int(current_edge[slot]) == int(robot.current_index):
                continue
            return True
        return False

    def _get_local_node_index(self, robot, next_location):
        if robot.node_coords is None or len(robot.node_coords) == 0:
            return int(robot.current_index)
        exact = np.argwhere(np.all(robot.node_coords == next_location, axis=1))
        if exact.size > 0:
            return int(exact[0][0])
        distances = np.linalg.norm(robot.node_coords - next_location, axis=1)
        return int(np.argmin(distances))

    def run_episode(self):
        team_node_managers = [robot.node_manager for robot in self.robot_list]
        for robot in self.robot_list:
            robot.update_graph(self.env.get_agent_map_info(robot.id), self.env.robot_locations[robot.id].copy())
        for robot in self.robot_list:
            robot.update_planning_state()
        self._update_merged_graph()
        self._update_objective_state()
        for robot in self.robot_list:
            self._set_agent_budget_context(robot)

        mission_failure = False
        max_decision_steps = MAX_EPISODE_STEP + self._budget_to_decision_steps(np.max(self.initial_budgets))

        for i in range(max_decision_steps):
            selected_locations = [robot.location.copy() for robot in self.robot_list]
            dist_list = [0.0 for _ in range(self.n_agents)]
            next_heading_index_list = [self._get_heading_index_towards(robot, robot.location) for robot in self.robot_list]
            observations = {}
            active_explorer_ids = []
            merged_was_completed = self.merged_objective_completed

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
                    self._close_agent_replay(robot, team_node_managers)
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
                    self._close_agent_replay(robot, team_node_managers)
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
                next_location, _, _, next_heading_index = robot.select_next_waypoint(observation)
                if not self._exploration_move_preserves_return_budget(robot, next_location):
                    self.returning_agents[robot_id] = True
                    self._close_agent_replay(robot, team_node_managers)
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

            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                if solved_locations.size == 0:
                    continue
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                    robot_id = arriving_sequence[j]
                    if robot_id in active_explorer_ids:
                        replacement_location, replacement_heading_index = self._get_collision_free_candidate(
                            self.robot_list[robot_id],
                            solved_locations,
                        )
                        if replacement_location is None:
                            replacement_location = self.robot_list[robot_id].location.copy()
                            replacement_heading_index = self._get_heading_index_towards(self.robot_list[robot_id], replacement_location)
                    else:
                        replacement_location = self.robot_list[robot_id].location.copy()
                        replacement_heading_index = self._get_heading_index_towards(self.robot_list[robot_id], replacement_location)

                    selected_location = replacement_location
                    next_heading_index_list[robot_id] = replacement_heading_index
                    selected_locations_in_arriving_sequence[j] = selected_location
                    selected_locations[robot_id] = selected_location
                    break

            still_active_explorer_ids = []
            selected_locations_list = [selected_locations[k].copy() for k in range(self.n_agents)]
            for robot_id in active_explorer_ids:
                robot = self.robot_list[robot_id]
                if self._exploration_move_preserves_return_budget(robot, selected_locations[robot_id]):
                    still_active_explorer_ids.append(robot_id)
                    continue

                self.returning_agents[robot_id] = True
                self._close_agent_replay(robot, team_node_managers)
                if self._set_return_target(robot, selected_locations_list, dist_list, next_heading_index_list):
                    mission_failure = True

            if mission_failure:
                break

            active_explorer_ids = still_active_explorer_ids
            selected_locations = np.asarray(selected_locations_list).reshape(-1, 2)

            if len(active_explorer_ids) == 0 and np.all(self.returning_agents):
                if not SIMULATE_RETURN_TO_BASE or self._all_agents_at_base():
                    break

            for robot_id in active_explorer_ids:
                robot = self.robot_list[robot_id]
                observation = observations[robot_id]
                robot.save_observation(observation)
                if self.use_merged_critic:
                    critic_observation = self.merged_critic_manager.get_critic_observation(
                        robot.location,
                        robot.id,
                        self.env.robot_locations,
                        team_node_managers,
                        self.env.get_agent_map_info(robot.id),
                        local_observation=observation,
                        local_node_coords=robot.node_coords,
                        base_location=self.base_locations[robot.id],
                    )
                    robot.current_critic_index = critic_observation[3][0, 0, 0].item()
                    robot.save_critic_observation(critic_observation)
                else:
                    ground_truth_observation = robot.ground_truth_node_manager.get_ground_truth_observation(robot.location, base_location=self.base_locations[robot.id])
                    robot.save_ground_truth_observation(ground_truth_observation)

            robot_locations_sim = []
            robot_headings_sim = []
            executed_action_index_list = {}
            for k, (robot, next_location, next_heading_index) in enumerate(zip(self.robot_list, selected_locations, next_heading_index_list)):
                robot_current_cell = get_cell_position_from_coords(robot.location, self.env.belief_info)
                robot_cell = get_cell_position_from_coords(next_location, self.env.belief_info)

                next_heading = next_heading_index*(360/NUM_ANGLES_BIN)
                final_heading = compute_allowable_heading(robot.location, next_location, robot.heading, next_heading, robot.velocity, robot.yaw_rate)
                if k in active_explorer_ids:
                    executed_action_index_list[k] = robot.get_executed_action_index(next_location, final_heading)

                # Generate intermediate points
                intermediate_cells = np.linspace(robot_current_cell, robot_cell, self.sim_steps+1)[1:] 

                # Round to nearest integer to get valid cell coordinates
                intermediate_cells = np.round(intermediate_cells).astype(int)
                intermediate_headings = self.smooth_heading_change(robot.heading, final_heading, steps=self.sim_steps)

                robot_locations_sim.append(intermediate_cells)
                robot_headings_sim.append(intermediate_headings)
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
                    robot_location_sim_step.append(robot_locations_sim[q][l])
                    robot_heading_sim_step.append(robot_headings_sim[q][l])
                self.env.refresh_merged_belief()
                
                if self.save_image:
                    num_frame = i * self.sim_steps + l
                    self.plot_local_env_sim(num_frame, robot_location_sim_step, robot_heading_sim_step, locations_are_cells=True)

            # Apply all final positions before reward computation to avoid order-dependent rewards.
            previous_locations = [robot.location.copy() for robot in self.robot_list]
            for robot, next_location in zip(self.robot_list, selected_locations):
                self.env.final_sim_step(next_location, robot.id)
                traveled_distance = float(np.linalg.norm(previous_locations[robot.id] - next_location))
                if traveled_distance > 0.0:
                    self.remaining_budgets[robot.id] -= traveled_distance

            reward_list = []
            # Collect robot headings for overlap reward calculation
            robot_headings_list = [robot.heading for robot in self.robot_list]

            for robot, next_location in zip(self.robot_list, selected_locations):
                self._append_trajectory_step(robot, next_location)

            for robot_id in active_explorer_ids:
                robot = self.robot_list[robot_id]
                next_location = selected_locations[robot_id]
                executed_action_index = executed_action_index_list[robot_id]

                node = robot.node_manager.nodes_dict.find((next_location[0], next_location[1])).data
                observable_frontiers = node.observable_frontiers
                observable_frontiers = np.array(list(observable_frontiers))
                if observable_frontiers.shape[0] > 0:

                    coords = np.array(node.coords)

                    delta = observable_frontiers - coords
                    angles = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]) % (2 * np.pi))

                    angle_diff = (angles - robot.heading + 180) % 360 - 180
                    current_observable_frontiers = observable_frontiers[np.abs(angle_diff) <= robot.fov / 2]

                    utility_reward = len(current_observable_frontiers) / ((2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE) / (360/robot.fov))

                else:
                    utility_reward = 0

                merged_node_utility = self.merged_map_manager.get_node_utility(next_location)
                merged_node_utility_reward = MERGED_NODE_UTILITY_REWARD_WEIGHT * (
                    merged_node_utility / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
                )

                preferred_angle = node.highest_utility_angle
                if preferred_angle == -360:
                    angle_reward = 0
                else:
                    angle_reward = np.cos(np.radians(robot.heading - preferred_angle))

                trajectory_angle = np.degrees(np.arctan2(next_location[1] - robot.location[1],
                                               next_location[0] - robot.location[0]) % (2 * np.pi))
                trajectory_reward = np.cos(np.radians(robot.heading - trajectory_angle))

                # Calculate overlap penalty for this agent
                # overlap_penalty = robot.calculate_overlap_reward(
                #     next_location,
                #     selected_locations,
                #     robot_headings_list
                # )

                low_utility_signal = 0.5 * utility_reward + merged_node_utility_reward
                if low_utility_signal < LOW_UTILITY_MOVE_THRESHOLD:
                    self.low_utility_streaks[robot_id] += 1
                else:
                    self.low_utility_streaks[robot_id] = 0

                # Penalise consecutive low-utility moves (streak length > 1 = oscillation pattern).
                repeated_low_utility_penalty = REPEATED_LOW_UTILITY_PENALTY * max(self.low_utility_streaks[robot_id] - 1, 0)

                # Scale the teammate-trail penalty by the decayed recency signal.
                trajectory_history_penalty = 0.15 * float(np.clip(node.visited_by_others, 0.0, 1.0))

                # budget_fraction = self.remaining_budgets[robot_id] / max(self.initial_budgets[robot_id], 1)
                # time_pressure = -TIME_PRESSURE_WEIGHT * (1.0 - budget_fraction)

                reward_list.append(
                    utility_reward
                    + merged_node_utility_reward
                    + trajectory_reward
                    - trajectory_history_penalty
                    - repeated_low_utility_penalty)
                robot.save_action(torch.tensor([executed_action_index], device=self.device))
            for robot in self.robot_list:
                robot.update_graph(self.env.get_agent_map_info(robot.id), self.env.robot_locations[robot.id].copy())
            for robot in self.robot_list:
                robot.mark_nodes_visited_by_others(self.env.robot_locations, self.trajectory_buffer)
                robot.update_planning_state()
            self._update_merged_graph()
            merged_completed_this_step = self._update_objective_state()
            for robot in self.robot_list:
                self._set_agent_budget_context(robot)

            if np.any(self.remaining_budgets < 0):
                mission_failure = True

            raw_team_reward = self.env.calculate_team_reward() - 0.5
            team_reward = 0.0 if merged_was_completed else raw_team_reward
            if merged_completed_this_step:
                team_reward += MERGED_SUCCESS_BONUS

            curr_node_indices = np.array([robot.current_index for robot in self.robot_list])
            curr_critic_indices = None
            if self.use_merged_critic and USE_COMMUNICATION:
                curr_critic_indices = np.array([self.merged_critic_manager.get_node_index(robot.location) for robot in self.robot_list])

            for robot_id, reward in zip(active_explorer_ids, reward_list):
                robot = self.robot_list[robot_id]
                local_completed = self._local_objective_completed(robot)
                individual_completion_bonus = 0.0
                if local_completed and not self.individual_objective_completed[robot_id]:
                    self.individual_objective_completed[robot_id] = True
                    individual_completion_bonus = INDIVIDUAL_SUCCESS_BONUS
                robot_should_return = (
                    local_completed
                    or self._should_force_return(robot.get_distance_to_base(), self.remaining_budgets[robot_id])
                )
                transition_done = mission_failure or robot_should_return
                robot.save_reward(reward + team_reward + individual_completion_bonus)
                if USE_COMMUNICATION:
                    if self.use_merged_critic:
                        robot.save_all_indices(curr_critic_indices)
                    else:
                        robot.save_all_indices(curr_node_indices)
                robot.save_done(transition_done)
                if robot_should_return:
                    self.returning_agents[robot_id] = True
                    self._close_agent_replay(robot, team_node_managers)

            if mission_failure:
                break

            if np.all(self.returning_agents):
                if not SIMULATE_RETURN_TO_BASE or self._all_agents_at_base():
                    break

        self._finalize_open_agent_replays(team_node_managers)
        for robot in self.robot_list:
            for buffer_idx in range(len(self.episode_buffer)):
                self.episode_buffer[buffer_idx] += robot.episode_buffer[buffer_idx]

        final_travel_dist = self._get_max_travel_dist()
        self.perf_metrics['travel_dist'] = final_travel_dist
        self.perf_metrics['merged_travel_dist'] = (
            self.merged_completion_travel_dist
            if self.merged_completion_travel_dist is not None else final_travel_dist
        )
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = self._episode_success(mission_failure)

        # Compute actual mean per-step reward from this episode's replay buffer
        reward_entries = self.episode_buffer[9]  # list of reward tensors
        if len(reward_entries) > 0:
            self.perf_metrics['episode_reward'] = float(
                torch.stack(reward_entries).mean().item()
            )
        else:
            self.perf_metrics['episode_reward'] = 0.0

        # Save episode video.
        if self.save_image:
            make_video(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def smooth_heading_change(self, prev_heading, heading, steps=10):
        # Ensure both angles are in the range [0, 360)
        prev_heading = prev_heading % 360
        heading = heading % 360

        # Calculate the angle difference
        diff = heading - prev_heading
        
        # Adjust for the shortest path
        if abs(diff) > 180:
            diff = diff - 360 if diff > 0 else diff + 360

        # Generate intermediate angles
        intermediate_headings = [
            (prev_heading + i * diff / steps) % 360
            for i in range(1, steps)
        ]

        # Ensure the final heading is exactly the target heading
        intermediate_headings.append(heading)
        return intermediate_headings
            
    def heading_to_vector(self, heading, length=25):
        # Convert heading to vector
        if isinstance(heading, (list, np.ndarray)):
            heading = heading[0]
        heading_rad = np.radians(heading)
        return np.cos(heading_rad) * length, np.sin(heading_rad) * length

    @staticmethod
    def _build_plot_updating_map(robot, map_info, location):
        updating_map_origin_x = (location[0] - robot.updating_map_size / 2)
        updating_map_origin_y = (location[1] - robot.updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + robot.updating_map_size
        updating_map_top_y = updating_map_origin_y + robot.updating_map_size

        min_x = map_info.map_origin_x
        min_y = map_info.map_origin_y
        max_x = map_info.map_origin_x + robot.cell_size * (map_info.map.shape[1] - 1)
        max_y = map_info.map_origin_y + robot.cell_size * (map_info.map.shape[0] - 1)

        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        updating_map_origin_x = (updating_map_origin_x // robot.cell_size + 1) * robot.cell_size
        updating_map_origin_y = (updating_map_origin_y // robot.cell_size + 1) * robot.cell_size
        updating_map_top_x = (updating_map_top_x // robot.cell_size) * robot.cell_size
        updating_map_top_y = (updating_map_top_y // robot.cell_size) * robot.cell_size

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, map_info)

        updating_map = map_info.map[
            updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1] + 1,
            updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0] + 1,
        ]
        return MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, robot.cell_size)

    @classmethod
    def _build_plot_node_manager(cls, robot, map_info, location):
        plot_node_manager = deepcopy(robot.node_manager)
        updating_map_info = cls._build_plot_updating_map(robot, map_info, location)
        frontiers = get_frontier_in_map(updating_map_info)
        plot_node_manager.update_graph(location, frontiers, updating_map_info, map_info)
        return plot_node_manager

    @staticmethod
    def _get_node_plot_data(node_manager):
        node_coords = []
        node_utility = []
        for node in node_manager.nodes_dict.__iter__():
            node_coords.append(node.data.coords)
            node_utility.append(node.data.utility)
        if not node_coords:
            return None, None
        return np.asarray(node_coords).reshape(-1, 2), np.asarray(node_utility)

    @staticmethod
    def _draw_node_manager_edges(ax, node_manager, map_info, color='#5a5a5a', linewidth=0.7, alpha=0.45):
        drawn_edges = set()
        for node in node_manager.nodes_dict.__iter__():
            start_key = tuple(np.round(node.data.coords, 1).tolist())
            start_cell = get_cell_position_from_coords(np.asarray(start_key), map_info)
            for neighbor_coords in node.data.neighbor_list[1:]:
                nb_key = tuple(np.round(neighbor_coords, 1).tolist())
                edge_key = (start_key, nb_key) if start_key < nb_key else (nb_key, start_key)
                if edge_key in drawn_edges:
                    continue
                drawn_edges.add(edge_key)
                end_cell = get_cell_position_from_coords(np.asarray(nb_key), map_info)
                ax.plot(
                    [start_cell[0], end_cell[0]],
                    [start_cell[1], end_cell[1]],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=1.8,
                )

    def _draw_robot_overlay(self, ax, location, heading, color, sensing_range, draw_fov=True, linewidth=1.2):
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

    def _get_agent_explored_rates(self):
        total_free = np.sum(self.env.ground_truth == FREE)
        rates = []
        for agent_id in range(self.n_agents):
            agent_map = self.env.get_agent_map_info(agent_id).map
            rate = np.sum(agent_map == FREE) / total_free if total_free > 0 else 0
            rates.append(rate)
        return rates

    def _locations_to_plot_cells(self, robot_locations, map_info, locations_are_cells):
        robot_locations = np.asarray(robot_locations)
        if locations_are_cells:
            return robot_locations
        return get_cell_position_from_coords(robot_locations, map_info)

    def plot_local_env_sim(self, step, robot_locations, robot_headings, locations_are_cells=False):
        plt.switch_backend('agg')
        n_cols = max(4, self.n_agents)
        fig = plt.figure(figsize=(15, 7.5), constrained_layout=True)
        gs = fig.add_gridspec(2, n_cols, height_ratios=[1.0, 1.05])
        color_list = ['tab:red', 'tab:blue', 'tab:green', 'goldenrod', 'tab:purple', 'tab:brown']
        color_name = ['Red', 'Blue', 'Green', 'Yellow', 'Purple', 'Brown']
        sensing_range = self.sensor_range / CELL_SIZE
        plot_robot_locations = self._locations_to_plot_cells(robot_locations, self.env.belief_info, locations_are_cells)
        plot_robot_coords = get_coords_from_cell_position(plot_robot_locations, self.env.belief_info).reshape(-1, 2)
        total_free = np.sum(self.env.ground_truth == FREE)
        agent_map_infos = [self.env.get_agent_map_info(robot.id) for robot in self.robot_list]
        agent_plot_node_managers = [
            self._build_plot_node_manager(robot, agent_map_info, plot_location)
            for robot, agent_map_info, plot_location in zip(self.robot_list, agent_map_infos, plot_robot_coords)
        ]
        agent_plot_node_data = [self._get_node_plot_data(node_manager) for node_manager in agent_plot_node_managers]

        merged_ax = fig.add_subplot(gs[0, 0])
        merged_ax.imshow(self.env.robot_belief, cmap='gray', interpolation='nearest', vmin=OCCUPIED, vmax=FREE)
        merged_ax.axis('off')
        merged_ax.set_title('Merged Team Belief', fontsize=10, fontweight='bold')
        xlim = merged_ax.get_xlim()
        ylim = merged_ax.get_ylim()
        merged_ax.set_xlim(xlim[0], xlim[1])
        merged_ax.set_ylim(ylim[0], ylim[1])

        global_frontiers = get_frontier_in_map(self.env.belief_info)
        if len(global_frontiers) != 0:
            frontier_cells = get_cell_position_from_coords(np.array(list(global_frontiers)), self.env.belief_info)
            if len(global_frontiers) == 1:
                frontier_cells = frontier_cells.reshape(1, 2)
            merged_ax.scatter(frontier_cells[:, 0], frontier_cells[:, 1], s=2, c='r', zorder=4)

        merged_node_coords, merged_node_utility = self._get_node_plot_data(self.merged_map_manager.node_manager)
        if merged_node_coords is not None:
            self._draw_node_manager_edges(merged_ax, self.merged_map_manager.node_manager, self.env.belief_info)
            nodes = get_cell_position_from_coords(merged_node_coords, self.env.belief_info)
            merged_ax.scatter(nodes[:, 0], nodes[:, 1], c=merged_node_utility, s=8, zorder=2)
            for i, (x, y) in enumerate(nodes):
                merged_ax.text(x - 3, y - 3, f'{merged_node_utility[i]:.0f}', ha='center', va='bottom', fontsize=3, color='blue', zorder=3)

        for robot, location, heading in zip(self.robot_list, plot_robot_locations, robot_headings):
            c = color_list[robot.id % len(color_list)]
            self._draw_robot_overlay(merged_ax, location, heading, c, sensing_range, draw_fov=True, linewidth=1.2)

        fov_ax = fig.add_subplot(gs[0, 1])
        fov_ax.imshow(self.env.robot_belief, cmap='gray', interpolation='nearest', vmin=OCCUPIED, vmax=FREE)
        fov_ax.axis('off')
        fov_ax.set_xlim(xlim[0], xlim[1])
        fov_ax.set_ylim(ylim[0], ylim[1])
        fov_ax.set_title('Team Motion + FoV', fontsize=10, fontweight='bold')

        if len(global_frontiers) != 0:
            fov_ax.scatter(frontier_cells[:, 0], frontier_cells[:, 1], s=2, c='r', zorder=4)

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
            self._draw_robot_overlay(fov_ax, location, heading, c, sensing_range, draw_fov=True, linewidth=1.2)

        gt_ax = fig.add_subplot(gs[0, 2])
        gt_ax.imshow(
            self.robot_list[0].ground_truth_node_manager.ground_truth_map_info.map,
            cmap='gray',
            interpolation='nearest',
            vmin=OCCUPIED,
            vmax=FREE,
        )
        gt_ax.set_xlim(xlim[0], xlim[1])
        gt_ax.set_ylim(ylim[0], ylim[1])
        gt_ax.axis('off')
        gt_ax.set_title('Ground Truth', fontsize=10, fontweight='bold')

        if self.robot_list[0].ground_truth_node_manager.ground_truth_node_coords is not None:
            nodes = get_cell_position_from_coords(
                self.robot_list[0].ground_truth_node_manager.ground_truth_node_coords,
                self.robot_list[0].ground_truth_node_manager.ground_truth_map_info,
            )
            gt_ax.scatter(nodes[:, 0], nodes[:, 1], c=self.robot_list[0].ground_truth_node_manager.explored_sign, s=8, zorder=2)

        for i, (location, heading) in enumerate(zip(plot_robot_locations, robot_headings)):
            c = color_list[i % len(color_list)]
            self._draw_robot_overlay(gt_ax, location, heading, c, sensing_range, draw_fov=True, linewidth=1.2)

        summary_ax = fig.add_subplot(gs[0, 3])
        summary_ax.axis('off')
        summary_ax.set_xlim(0, 1)
        summary_ax.set_ylim(0, 1)

        card = Rectangle(
            (0.01, 0.01),
            0.98,
            0.98,
            facecolor='#f7f5ef',
            edgecolor='#c8c2b5',
            linewidth=1.4,
            clip_on=False,
        )
        summary_ax.add_patch(card)
        summary_ax.text(0.08, 0.96, 'Legend & Stats', fontsize=11.5, fontweight='bold', color='#2b2b2b', va='top')

        legend_rows = [
            [('Occupied', '#101010'), ('Unknown', '#7f7f7f')],
            [('Free', '#f5f5f5'), ('Utility Node', 'orange')],
            [('Frontier', 'red'), None],
        ]

        legend_col_x = [0.08, 0.53]
        legend_row_y = [0.84, 0.77, 0.70]
        for row_y, row_items in zip(legend_row_y, legend_rows):
            for col_x, item in zip(legend_col_x, row_items):
                if item is None:
                    continue
                label, facecolor = item
                swatch = Rectangle(
                    (col_x, row_y - 0.020),
                    0.055,
                    0.034,
                    facecolor=facecolor,
                    edgecolor='#333333',
                    linewidth=0.8,
                )
                summary_ax.add_patch(swatch)
                summary_ax.text(
                    col_x + 0.09,
                    row_y - 0.003,
                    label,
                    fontsize=9.0,
                    fontweight='bold',
                    color='#2b2b2b',
                    va='center',
                )

        agent_explored_rates = self._get_agent_explored_rates()
        avg_agent_explored_rate = float(np.mean(agent_explored_rates))

        summary_ax.plot([0.08, 0.92], [0.61, 0.61], color='#d8d2c7', linewidth=1.0)
        summary_ax.text(0.08, 0.58, 'Episode', fontsize=10, fontweight='bold', color='#2b2b2b', va='top')
        summary_ax.text(0.08, 0.49, 'Merged explored', fontsize=9.0, fontweight='bold', color='#4a4a4a', va='center')
        summary_ax.text(0.08, 0.41, 'Avg agent explored', fontsize=9.0, fontweight='bold', color='#4a4a4a', va='center')
        summary_ax.text(0.08, 0.33, 'Max travel dist', fontsize=9.0, fontweight='bold', color='#4a4a4a', va='center')
        summary_ax.text(
            0.86,
            0.49,
            f'{self.env.explored_rate:.1%}',
            fontsize=9.2,
            fontweight='bold',
            color='#2b2b2b',
            va='center',
            ha='right',
        )
        summary_ax.text(
            0.86,
            0.41,
            f'{avg_agent_explored_rate:.1%}',
            fontsize=9.2,
            fontweight='bold',
            color='#2b2b2b',
            va='center',
            ha='right',
        )
        summary_ax.text(
            0.86,
            0.33,
            f'{max([robot.travel_dist for robot in self.robot_list]):.1f}',
            fontsize=9.2,
            fontweight='bold',
            color='#2b2b2b',
            va='center',
            ha='right',
        )

        summary_ax.plot([0.08, 0.92], [0.29, 0.29], color='#d8d2c7', linewidth=1.0)
        summary_ax.text(0.08, 0.26, 'Per-Agent', fontsize=10, fontweight='bold', color='#2b2b2b', va='top')
        summary_ax.text(0.62, 0.20, 'Explored', fontsize=8.6, fontweight='bold', color='#6a6a6a', va='center', ha='right')
        summary_ax.text(0.86, 0.20, 'Nodes', fontsize=8.6, fontweight='bold', color='#6a6a6a', va='center', ha='right')
        row_y = 0.155
        for robot, agent_explored_rate, plot_node_data in zip(self.robot_list, agent_explored_rates, agent_plot_node_data):
            color_idx = robot.id % len(color_list)
            plot_node_coords, _ = plot_node_data
            num_nodes = 0 if plot_node_coords is None else len(plot_node_coords)
            summary_ax.plot([0.08, 0.13], [row_y, row_y], color=color_list[color_idx], linewidth=3.5, solid_capstyle='round')
            summary_ax.text(
                0.17,
                row_y,
                color_name[color_idx],
                fontsize=9.0,
                fontweight='bold',
                color='#2b2b2b',
                va='center',
            )
            summary_ax.text(
                0.62,
                row_y,
                f'{agent_explored_rate:.1%}',
                fontsize=9.0,
                fontweight='bold',
                color='#2b2b2b',
                va='center',
                ha='right',
            )
            summary_ax.text(
                0.86,
                row_y,
                f'{num_nodes}',
                fontsize=9.0,
                fontweight='bold',
                color='#2b2b2b',
                va='center',
                ha='right',
            )
            row_y -= 0.04

        for robot, agent_explored_rate, plot_node_manager, plot_node_data in zip(
            self.robot_list,
            agent_explored_rates,
            agent_plot_node_managers,
            agent_plot_node_data,
        ):
            agent_ax = fig.add_subplot(gs[1, robot.id])
            color_idx = robot.id % len(color_list)
            c = color_list[color_idx]
            agent_map_info = agent_map_infos[robot.id]
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
                (np.array(trajectory_x) - self.env.belief_info.map_origin_x) / self.env.belief_info.cell_size,
                (np.array(trajectory_y) - self.env.belief_info.map_origin_y) / self.env.belief_info.cell_size,
                c,
                linewidth=1.5,
                alpha=0.9,
                zorder=2,
            )

            plot_node_coords, plot_node_utility = plot_node_data
            if plot_node_coords is not None and len(plot_node_coords) > 0:
                nodes = get_cell_position_from_coords(plot_node_coords, agent_map_info)
                if nodes.ndim == 1:
                    nodes = nodes.reshape(1, 2)
                agent_ax.scatter(nodes[:, 0], nodes[:, 1], c=c, s=8, zorder=3, alpha=0.65)
                utility_mask = plot_node_utility > 0
                if np.any(utility_mask):
                    agent_ax.scatter(
                        nodes[utility_mask, 0],
                        nodes[utility_mask, 1],
                        c='orange',
                        s=20,
                        zorder=4,
                        alpha=0.8,
                    )

            agent_frontiers = get_frontier_in_map(agent_map_info)
            if len(agent_frontiers) != 0:
                frontier_cells = get_cell_position_from_coords(np.array(list(agent_frontiers)), agent_map_info)
                if len(agent_frontiers) == 1:
                    frontier_cells = frontier_cells.reshape(1, 2)
                agent_ax.scatter(frontier_cells[:, 0], frontier_cells[:, 1], s=2, c='r', zorder=5)

            location = plot_robot_locations[robot.id]
            heading = robot_headings[robot.id]
            self._draw_robot_overlay(agent_ax, location, heading, c, sensing_range, draw_fov=True, linewidth=1.2)

            num_nodes = 0 if plot_node_coords is None else len(plot_node_coords)
            agent_ax.set_title(
                f'{color_name[color_idx]} Belief\nExplored: {agent_explored_rate:.1%}  Nodes: {num_nodes}',
                fontsize=9,
                fontweight='bold',
                color=c,
            )

            # Budget loading bar (below each agent subplot), expressed in meters.
            initial_budget = max(float(self.initial_budgets[robot.id]), 1.0) if hasattr(self, 'initial_budgets') else max(float(BUDGET), 1.0)
            remaining = float(max(self.remaining_budgets[robot.id], 0.0)) if hasattr(self, 'remaining_budgets') else float(BUDGET)
            fraction_remaining = remaining / initial_budget
            if fraction_remaining > 0.5:
                bar_color = '#2ecc71'   # green  — high budget
            elif fraction_remaining > 0.25:
                bar_color = '#f39c12'  # orange — moderate budget
            else:
                bar_color = '#e74c3c'  # red    — low budget
            bar_bg = Rectangle((0.0, 0.955), 1.0, 0.04,
                                transform=agent_ax.transAxes,
                                color='#d0d0d0', clip_on=True, zorder=15)
            agent_ax.add_patch(bar_bg)
            bar_fg = Rectangle((0.0, 0.955), fraction_remaining, 0.04,
                                transform=agent_ax.transAxes,
                                color=bar_color, clip_on=True, zorder=16)
            agent_ax.add_patch(bar_fg)
            agent_ax.text(
                0.5, 0.975,
                f'{remaining:.1f}/{initial_budget:.1f} m{" | return" if hasattr(self, "returning_agents") and self.returning_agents[robot.id] else ""}',
                transform=agent_ax.transAxes,
                fontsize=6.5, ha='center', va='center',
                fontweight='bold', color='#1a1a1a',
                clip_on=True, zorder=17,
            )

        for empty_col in range(self.n_agents, n_cols):
            empty_ax = fig.add_subplot(gs[1, empty_col])
            empty_ax.axis('off')

        robot_headings_str = [f"{color_name[robot.id % len(color_name)]} {robot.heading:.0f} deg" for robot in self.robot_list]
        fig.suptitle(
            'Experiment ID: {}\nRobot headings: {}'.format(
                FOLDER_NAME,
                ', '.join(robot_headings_str)
            ),
            fontweight='bold',
            fontsize=11,
        )
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)

if __name__ == '__main__':
    from parameter import *
    import torch

    def load_compatible_state_dict(module, checkpoint_state, label):
        model_state = module.state_dict()
        compatible_state = {
            key: value for key, value in checkpoint_state.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        missing_or_mismatched = len(model_state) - len(compatible_state)
        module.load_state_dict(compatible_state, strict=False)
        print(f'Loaded {label}: {len(compatible_state)} tensors, skipped {missing_or_mismatched}')

    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN)
    if LOAD_MODEL:
        checkpoint = torch.load(load_path + '/checkpoint.pth', map_location='cpu')
        load_compatible_state_dict(policy_net, checkpoint['policy_model'], 'policy')
        print('Policy loaded!')
    worker = MultiAgentWorker(0, policy_net, 888, 'cpu', True)
    worker.run_episode()
