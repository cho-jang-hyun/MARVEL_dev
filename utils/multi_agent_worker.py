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
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
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

    def run_episode(self):
        done = False
        team_node_managers = [robot.node_manager for robot in self.robot_list]
        for robot in self.robot_list:
            robot.update_graph(self.env.get_agent_map_info(robot.id), self.env.robot_locations[robot.id].copy())
        for robot in self.robot_list:
            robot.update_planning_state()
        self.merged_map_manager.update_graph(self.env.belief_info, self.env.robot_locations)

        for i in range(MAX_EPISODE_STEP):

            selected_locations = []
            dist_list = []
            next_heading_index_list = []
            for robot in self.robot_list:
                observation = robot.get_observation(
                    robot_locations=self.env.robot_locations,
                    trajectory_buffer=self.trajectory_buffer
                )

                robot.save_observation(observation)
                if self.use_merged_critic:
                    critic_observation = self.merged_critic_manager.get_critic_observation(
                        robot.location,
                        team_node_managers,
                        self.env.get_agent_map_info(robot.id),
                        local_observation=observation,
                        local_node_coords=robot.node_coords,
                    )
                    robot.current_critic_index = critic_observation[3][0, 0, 0].item()
                    robot.save_critic_observation(critic_observation)
                else:
                    ground_truth_observation = robot.ground_truth_node_manager.get_ground_truth_observation(robot.location)
                    robot.save_ground_truth_observation(ground_truth_observation)

                next_location, _, _, next_heading_index = robot.select_next_waypoint(observation)

                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))
                next_heading_index_list.append(next_heading_index)

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

            # Solve collision
            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                    id = arriving_sequence[j]
                    replacement_location, replacement_heading_index = self._get_collision_free_candidate(
                        self.robot_list[id],
                        solved_locations,
                    )
                    if replacement_location is None:
                        break

                    selected_location = replacement_location
                    next_heading_index_list[id] = replacement_heading_index

                    selected_locations_in_arriving_sequence[j] = selected_location
                    selected_locations[id] = selected_location


            # Compute simulation data
            robot_locations_sim = []
            robot_headings_sim = []
            all_robots_heading_list = []
            executed_action_index_list = []
            executed_next_node_index_list = []
            executed_critic_next_node_index_list = []
            for k, (robot, next_location, next_heading_index) in enumerate(zip(self.robot_list, selected_locations, next_heading_index_list)):
                robot_current_cell = get_cell_position_from_coords(robot.location, self.env.belief_info)
                robot_cell = get_cell_position_from_coords(next_location, self.env.belief_info)

                next_heading = next_heading_index*(360/NUM_ANGLES_BIN)
                final_heading = compute_allowable_heading(robot.location, next_location, robot.heading, next_heading, robot.velocity, robot.yaw_rate)
                executed_action_index = robot.get_executed_action_index(next_location, final_heading)

                # Generate intermediate points
                intermediate_cells = np.linspace(robot_current_cell, robot_cell, self.sim_steps+1)[1:] 

                # Round to nearest integer to get valid cell coordinates
                intermediate_cells = np.round(intermediate_cells).astype(int)
                intermediate_headings = self.smooth_heading_change(robot.heading, final_heading, steps=self.sim_steps)

                robot_locations_sim.append(intermediate_cells)
                robot_headings_sim.append(intermediate_headings)
                all_robots_heading_list.append(final_heading)
                executed_action_index_list.append(executed_action_index)
                executed_next_node_index_list.append(int(np.argwhere(np.all(robot.node_coords == next_location, axis=1))[0][0]))
                if self.use_merged_critic:
                    executed_critic_next_node_index_list.append(
                        self.merged_critic_manager.get_node_index(next_location)
                    )

                robot.update_heading(final_heading)

            for l in range(self.sim_steps):
                robot_location_sim_step = []
                robot_heading_sim_step = []
                for q in range(self.n_agents):
                    self.env.update_robot_belief(q, robot_locations_sim[q][l], robot_headings_sim[q][l])
                    robot_location_sim_step.append(robot_locations_sim[q][l])
                    robot_heading_sim_step.append(robot_headings_sim[q][l])
                
                if self.save_image:
                    num_frame = i * self.sim_steps + l
                    self.plot_local_env_sim(num_frame, robot_location_sim_step, robot_heading_sim_step, locations_are_cells=True)

            # Apply all final positions before reward computation to avoid order-dependent rewards.
            for robot, next_location in zip(self.robot_list, selected_locations):
                self.env.final_sim_step(next_location, robot.id)

            reward_list = []
            # Collect robot headings for overlap reward calculation
            robot_headings_list = [robot.heading for robot in self.robot_list]

            for robot, next_location, executed_action_index in zip(self.robot_list, selected_locations, executed_action_index_list):
                # Update trajectory buffer
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
                overlap_penalty = robot.calculate_overlap_reward(
                    next_location,
                    selected_locations,
                    robot_headings_list
                )

                # Explicit penalty for redundantly following a teammate's trail
                trajectory_history_penalty = 0.0
                if node.visited_by_others > 0.9:
                    trajectory_history_penalty = 0.15

                reward_list.append(
                    utility_reward
                    + merged_node_utility_reward
                    + trajectory_reward
                    - overlap_penalty
                    - trajectory_history_penalty
                )
                robot.save_action(torch.tensor([executed_action_index], device=self.device))

                robot.update_graph(self.env.get_agent_map_info(robot.id), self.env.robot_locations[robot.id].copy())
                # Mark nodes visited by other agents based on FoV detection
                robot.mark_nodes_visited_by_others(self.env.robot_locations, self.trajectory_buffer)
            self.merged_map_manager.update_graph(self.env.belief_info, self.env.robot_locations)

            # End the episode when total environment exploration exceeds the threshold.
            if self.env.explored_rate > SUCCESS_THRESHOLD:
                done = True

            team_reward = self.env.calculate_team_reward() - 0.5
            if done:
                team_reward += 10

            curr_node_indices = np.array([robot.current_index for robot in self.robot_list])
            curr_critic_indices = None
            if self.use_merged_critic and USE_COMMUNICATION:
                curr_critic_indices = np.array([robot.current_critic_index for robot in self.robot_list])
            for robot, reward in zip(self.robot_list, reward_list):
                robot.save_reward(reward + team_reward)
                # Only save all agent indices when communication is enabled
                # When USE_COMMUNICATION=False, agents rely solely on FOV-detected trajectories
                if USE_COMMUNICATION:
                    if self.use_merged_critic:
                        robot.save_all_indices(curr_critic_indices)
                    else:
                        robot.save_all_indices(curr_node_indices)
                robot.update_planning_state()
                robot.save_done(done)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save episode buffer
        for robot in self.robot_list:
            observation = robot.get_observation(
                robot_locations=self.env.robot_locations,
                trajectory_buffer=self.trajectory_buffer
            )
            joint_next_index_list = executed_next_node_index_list
            if self.use_merged_critic and USE_COMMUNICATION:
                joint_next_index_list = executed_critic_next_node_index_list
            robot.save_next_observations(observation, joint_next_index_list)
            if self.use_merged_critic:
                critic_observation = self.merged_critic_manager.get_critic_observation(
                    robot.location,
                    team_node_managers,
                    self.env.get_agent_map_info(robot.id),
                    local_observation=observation,
                    local_node_coords=robot.node_coords,
                )
                robot.save_next_critic_observations(critic_observation)
            else:
                ground_truth_observation = robot.ground_truth_node_manager.get_ground_truth_observation(robot.location)
                robot.save_next_ground_truth_observations(ground_truth_observation)
            for i in range(len(self.episode_buffer)):
                self.episode_buffer[i] += robot.episode_buffer[i]

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

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

    def _draw_robot_overlay(self, ax, location, heading, color, sensing_range, draw_fov=False, linewidth=1.5):
        dx, dy = self.heading_to_vector(heading, length=sensing_range)
        arrow = FancyArrowPatch(
            (location[0], location[1]),
            (location[0] + dx / 1.25, location[1] + dy / 1.25),
            mutation_scale=10,
            color=color,
            arrowstyle='-|>',
            linewidth=linewidth,
            zorder=12,
        )
        ax.add_artist(arrow)
        ax.plot(location[0], location[1], 'o', color=color, markersize=5, zorder=13)

        if draw_fov:
            cone = Wedge(
                center=(location[0], location[1]),
                r=self.sensor_range / CELL_SIZE,
                theta1=(heading - self.fov / 2),
                theta2=(heading + self.fov / 2),
                facecolor=color,
                alpha=0.5,
                linewidth=1.2,
                edgecolor=color,
                zorder=11,
            )
            ax.add_artist(cone)

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
        color_name = ['Red', 'Blue', 'Green', 'Yellow']
        sensing_range = self.sensor_range / CELL_SIZE
        plot_robot_locations = self._locations_to_plot_cells(robot_locations, self.env.belief_info, locations_are_cells)

        merged_ax = fig.add_subplot(gs[0, 0])
        merged_ax.imshow(self.env.robot_belief, cmap='gray', interpolation='nearest', vmin=OCCUPIED, vmax=FREE)
        merged_ax.axis('off')
        merged_ax.set_title('Merged Team Belief', fontsize=10, fontweight='bold')
        xlim = merged_ax.get_xlim()
        ylim = merged_ax.get_ylim()
        merged_ax.set_xlim(xlim[0], xlim[1])
        merged_ax.set_ylim(ylim[0], ylim[1])

        for robot, location, heading in zip(self.robot_list, plot_robot_locations, robot_headings):
            c = color_list[robot.id]
            self._draw_robot_overlay(merged_ax, location, heading, c, sensing_range, draw_fov=True, linewidth=1.2)

        fov_ax = fig.add_subplot(gs[0, 1])
        fov_ax.imshow(self.env.robot_belief, cmap='gray', interpolation='nearest', vmin=OCCUPIED, vmax=FREE)
        fov_ax.axis('off')
        fov_ax.set_xlim(xlim[0], xlim[1])
        fov_ax.set_ylim(ylim[0], ylim[1])
        fov_ax.set_title('Team Motion + FoV', fontsize=10, fontweight='bold')

        for robot, location, heading in zip(self.robot_list, plot_robot_locations, robot_headings):
            c = color_list[robot.id]
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

        for i, (location, heading) in enumerate(zip(plot_robot_locations, robot_headings)):
            c = color_list[i]
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
        legend_row_y = [0.84, 0.75, 0.66]
        for row_y, row_items in zip(legend_row_y, legend_rows):
            for col_x, item in zip(legend_col_x, row_items):
                if item is None:
                    continue
                label, facecolor = item
                swatch = Rectangle(
                    (col_x, row_y - 0.022),
                    0.055,
                    0.038,
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

        summary_ax.plot([0.08, 0.92], [0.58, 0.58], color='#d8d2c7', linewidth=1.0)
        summary_ax.text(0.08, 0.55, 'Episode', fontsize=10, fontweight='bold', color='#2b2b2b', va='top')
        summary_ax.text(0.08, 0.45, 'Merged explored', fontsize=9.0, fontweight='bold', color='#4a4a4a', va='center')
        summary_ax.text(0.08, 0.37, 'Max travel dist', fontsize=9.0, fontweight='bold', color='#4a4a4a', va='center')
        summary_ax.text(
            0.86,
            0.45,
            f'{self.env.explored_rate:.1%}',
            fontsize=9.2,
            fontweight='bold',
            color='#2b2b2b',
            va='center',
            ha='right',
        )
        summary_ax.text(
            0.86,
            0.37,
            f'{max([robot.travel_dist for robot in self.robot_list]):.1f}',
            fontsize=9.2,
            fontweight='bold',
            color='#2b2b2b',
            va='center',
            ha='right',
        )

        summary_ax.plot([0.08, 0.92], [0.33, 0.33], color='#d8d2c7', linewidth=1.0)
        summary_ax.text(0.08, 0.30, 'Per-Agent', fontsize=10, fontweight='bold', color='#2b2b2b', va='top')
        summary_ax.text(0.62, 0.24, 'Explored', fontsize=8.6, fontweight='bold', color='#6a6a6a', va='center', ha='right')
        summary_ax.text(0.86, 0.24, 'Nodes', fontsize=8.6, fontweight='bold', color='#6a6a6a', va='center', ha='right')
        row_y = 0.185
        for robot in self.robot_list:
            num_nodes = 0 if robot.node_coords is None else len(robot.node_coords)
            agent_map = self.env.agent_beliefs[robot.id]
            total_free = np.sum(self.env.ground_truth == FREE)
            agent_explored_rate = np.sum(agent_map == FREE) / total_free if total_free > 0 else 0
            summary_ax.plot([0.08, 0.13], [row_y, row_y], color=color_list[robot.id], linewidth=3.5, solid_capstyle='round')
            summary_ax.text(
                0.17,
                row_y,
                color_name[robot.id],
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
            row_y -= 0.045

        for robot in self.robot_list:
            agent_ax = fig.add_subplot(gs[1, robot.id])
            c = color_list[robot.id]
            agent_map_info = self.env.get_agent_map_info(robot.id)
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

            if robot.node_coords is not None and len(robot.node_coords) > 0:
                nodes = get_cell_position_from_coords(robot.node_coords, agent_map_info)
                if nodes.ndim == 1:
                    nodes = nodes.reshape(1, 2)
                agent_ax.scatter(nodes[:, 0], nodes[:, 1], c=c, s=8, zorder=3, alpha=0.65)
                utility_mask = robot.utility > 0
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

            num_nodes = 0 if robot.node_coords is None else len(robot.node_coords)
            total_free = np.sum(self.env.ground_truth == FREE)
            agent_explored_rate = np.sum(agent_map == FREE) / total_free if total_free > 0 else 0
            agent_ax.set_title(
                f'{color_name[robot.id]} Belief\nExplored: {agent_explored_rate:.1%}  Nodes: {num_nodes}',
                fontsize=9,
                fontweight='bold',
                color=c,
            )

        for empty_col in range(self.n_agents, n_cols):
            empty_ax = fig.add_subplot(gs[1, empty_col])
            empty_ax.axis('off')

        robot_headings_str = [f"{color_name[robot.id]} {robot.heading:.0f} deg" for robot in self.robot_list]
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
    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN)
    if LOAD_MODEL:
        checkpoint = torch.load(load_path + '/checkpoint.pth', map_location='cpu')
        policy_net.load_state_dict(checkpoint['policy_model'])
        print('Policy loaded!')
    worker = MultiAgentWorker(0, policy_net, 888, 'cpu', True)
    worker.run_episode()
