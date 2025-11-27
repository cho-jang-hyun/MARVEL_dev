import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.patches import Wedge, FancyArrowPatch
from shapely.geometry import Point, Polygon
from skimage.draw import polygon as sk_polygon
from collections import deque

from utils.env_test import Env
from utils.agent import Agent
from utils.utils import *
from utils.node_manager import NodeManager
from utils.ground_truth_node_manager import GroundTruthNodeManager
from utils.model import PolicyNet
from utils.motion_model import compute_allowable_heading
from test_parameter import *

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, n_agent, fov, sensor_range, utility_range, device='cpu', save_image=False, greedy=True):
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

        self.env = Env(global_step, self.fov, self.n_agents, self.sensor_range, plot=self.save_image)

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

    def run_episode(self):
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, self.env.robot_locations[robot.id].copy())
        for robot in self.robot_list:
            robot.update_planning_state()
        
        reach_checkpoint = False

        max_travel_dist = 0
        trajectory_length = 0

        length_history = [max_travel_dist]
        explored_rate_history = [self.env.explored_rate]
        overlap_rate = self.compute_overlap_rate(self.env.robot_locations, self.env.angles)
        overlap_ratio_history = [overlap_rate]

        setpoints = [[] for _ in range(self.n_agents)]
        headings = [[] for _ in range(self.n_agents)]


        for i in range(MAX_EPISODE_STEP):
            # print(' Current timestep: {}/{}'.format(i, MAX_EPISODE_STEP))
            selected_locations = []
            dist_list = []
            next_node_index_list = []
            next_heading_index_list = []
            for robot in self.robot_list:
                observation = robot.get_observation(
                    pad=False,
                    robot_locations=self.env.robot_locations,
                    trajectory_buffer=self.trajectory_buffer
                )

                next_location, next_node_index, _, next_heading_index = robot.select_next_waypoint(observation, greedy=self.greedy)

                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))
                next_node_index_list.append(next_node_index)
                next_heading_index_list.append(next_heading_index)

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

  
            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                    id = arriving_sequence[j]
                    nearby_nodes = self.robot_list[id].node_manager.nodes_dict.nearest_neighbors(
                        selected_location.tolist(), 25)
                    for node in nearby_nodes:
                        coords = node.data.coords
                        if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                            continue
                        selected_location = coords
                        break

                    selected_locations_in_arriving_sequence[j] = selected_location
                    selected_locations[id] = selected_location

            for p, location in enumerate(selected_locations):
                setpoints[p].append(location*self.scaling)

            # Compute simulation data
            robot_locations_sim = []
            robot_headings_sim = []
            all_robots_heading_list = []
            for k, (robot, next_location, next_heading_index) in enumerate(zip(self.robot_list, selected_locations, next_heading_index_list)):
                robot_current_cell = get_cell_position_from_coords(robot.location, self.env.belief_info)
                robot_cell = get_cell_position_from_coords(next_location, self.env.belief_info)

                next_heading = next_heading_index*(360/NUM_ANGLES_BIN)
                final_heading = compute_allowable_heading(robot.location, next_location, robot.heading, next_heading, robot.velocity, robot.yaw_rate)

                intermediate_cells = np.linspace(robot_current_cell, robot_cell, self.sim_steps+1)[1:] 

                intermediate_cells = np.round(intermediate_cells).astype(int)
                intermediate_headings = self.smooth_heading_change(robot.heading, final_heading, steps=self.sim_steps)

                robot_locations_sim.append(intermediate_cells)
                robot_headings_sim.append(intermediate_headings)
                all_robots_heading_list.append(final_heading)
                corrected_heading = self.correct_heading(final_heading)
                headings[k].append(corrected_heading)

                robot.update_heading(final_heading)

            for l in range(self.sim_steps):
                robot_location_sim_step = []
                robot_heading_sim_step = []
                for q in range(self.n_agents):
                    self.env.update_robot_belief(robot_locations_sim[q][l], robot_headings_sim[q][l])
                    robot_location_sim_step.append(robot_locations_sim[q][l])
                    robot_heading_sim_step.append(robot_headings_sim[q][l])
                
                if self.save_image:
                    num_frame = i * self.sim_steps + l
                    # Plot both the original view and individual agent views
                    self.plot_local_env_sim(num_frame, robot_location_sim_step, robot_heading_sim_step)
                    # Also create individual agent views to show decentralized learning
                    if num_frame % 5 == 0:  # Save individual views every 5 frames to avoid too many files
                        self.plot_individual_agent_views(num_frame, robot_location_sim_step, robot_heading_sim_step)

            for robot, next_location, next_node_index in zip(self.robot_list, selected_locations, next_node_index_list):
                self.env.final_sim_step(next_location, robot.id)

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

                robot.update_graph(self.env.belief_info, self.env.robot_locations[robot.id].copy())

            overlap_rate = self.compute_overlap_rate(selected_locations, all_robots_heading_list)

            for robot in self.robot_list:
                robot.update_planning_state()

            max_travel_dist += np.max(dist_list)
            length_history.append(max_travel_dist)
            explored_rate_history.append(self.env.explored_rate)
            overlap_ratio_history.append(overlap_rate)
            if self.env.explored_rate > INITIAL_EXPLORED_RATE and not reach_checkpoint:
                trajectory_length = max([robot.travel_dist for robot in self.robot_list])
                reach_checkpoint = True

            if self.env.explored_rate > 0.99:
                done = True

            if done:
                break

        # Save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        if trajectory_length > 0:
            self.perf_metrics['dist_to_0_90'] = trajectory_length
        else:
            self.perf_metrics['dist_to_0_90'] = []
        self.perf_metrics['length_history'] = length_history
        self.perf_metrics['explored_rate_history'] = explored_rate_history
        self.perf_metrics['overlap_ratio_history'] = overlap_ratio_history
    
        # Save gif
        if self.save_image:
            pass
            make_gif_test(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate, self.n_agents, self.fov, self.sensor_range)

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
    
    def create_sensing_mask(self, location, heading):
        mask = np.zeros_like(self.env.ground_truth)

        location_cell = get_cell_position_from_coords(location, self.env.belief_info)
        robot_point = Point(location_cell)

        start_angle = (heading - self.fov / 2 + 360) % 360
        end_angle = (heading + self.fov / 2) % 360

        sector_points = [robot_point]
        if start_angle <= end_angle:
            angle_range = np.linspace(start_angle, end_angle, 20)
        else:
            angle_range = np.concatenate([np.linspace(start_angle, 360, 10), np.linspace(0, end_angle, 10)])
        for angle in angle_range:  
            x = location_cell[0] + SENSOR_RANGE/CELL_SIZE * np.cos(np.radians(angle))
            y = location_cell[1] + SENSOR_RANGE/CELL_SIZE * np.sin(np.radians(angle))
            sector_points.append(Point(x, y))
        sector_points.append(robot_point)  
        sector = Polygon(sector_points)

        x_coords, y_coords = sector.exterior.xy
        y_coords = np.rint(y_coords).astype(int)
        x_coords = np.rint(x_coords).astype(int)
        rr, cc = sk_polygon(
                [int(round(y)) for y in y_coords],
                [int(round(x)) for x in x_coords],
                shape=mask.shape
            )
        
        free_connected_map = get_free_and_connected_map(location, self.env.belief_info)

        mask[rr, cc] = (free_connected_map[rr, cc] == free_connected_map[location_cell[1], location_cell[0]])
       
        return mask
    
    def compute_overlap_rate(self, all_robots_locations, robot_headings_list):
        all_robot_sensing_mask = []
        for robot_location, robot_heading in zip(all_robots_locations, robot_headings_list):
            robot_sensing_mask = self.create_sensing_mask(robot_location, robot_heading)
            all_robot_sensing_mask.append(robot_sensing_mask)
        
        total_mask = np.sum(all_robot_sensing_mask, axis=0)
        total_sensing_area = np.sum(total_mask > 0)
        total_overlap_area = np.sum(total_mask > 1)

        overlap_ratio = total_overlap_area / total_sensing_area  
        
        return overlap_ratio
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
            # Get agent's individual map belief
            agent_map = robot.map_info.map
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

        # Calculate layout: top row has 2 panels, bottom row has one panel per agent
        n_cols = max(2, self.n_agents)
        fig = plt.figure(figsize=(3 * n_cols, 6))

        color_list = ['r', 'b', 'g', 'y']
        color_name = ['Red', 'Blue', 'Green', 'Yellow']
        sensing_range = SENSOR_RANGE / CELL_SIZE

        # Detect robots in FOV for each robot
        fov_detections = {}
        for robot in self.robot_list:
            fov_detections[robot.id] = self.get_detected_robots_in_fov(robot, robot_locations, robot_headings)

        # Top row - Panel 1: Global belief map with trajectories
        plt.subplot(2, n_cols, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.title('Global Belief Map', fontsize=10, fontweight='bold')

        # First pass: Draw all trajectories (thinner, semi-transparent)
        for i, (robot, location, heading) in enumerate(zip(self.robot_list, robot_locations, robot_headings)):
            plot_id = robot.id % 4
            c = color_list[plot_id]
            robot_location = get_coords_from_cell_position(location, self.env.belief_info)
            trajectory_x = robot.trajectory_x.copy()
            trajectory_y = robot.trajectory_y.copy()
            trajectory_x.append(robot_location[0])
            trajectory_y.append(robot_location[1])
            plt.plot((np.array(trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
                     (np.array(trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size, c,
                     linewidth=1.2, alpha=0.4, zorder=1)

        # Second pass: Highlight detected trajectories and draw arrows
        for i, (robot, location, heading) in enumerate(zip(self.robot_list, robot_locations, robot_headings)):
            plot_id = robot.id % 4
            c = color_list[plot_id]
            dx, dy = self.heading_to_vector(heading, length=sensing_range)
            arrow = FancyArrowPatch((location[0], location[1]), (location[0] + dx/1.25, location[1] + dy/1.25),
                                    mutation_scale=10,
                                    color=c,
                                    arrowstyle='-|>')
            plt.gca().add_artist(arrow)

            # Highlight trajectory if this robot is detected by any other robot
            is_detected = False
            for detector_id, detected_list in fov_detections.items():
                if robot.id in detected_list:
                    is_detected = True
                    break

            if is_detected:
                robot_location = get_coords_from_cell_position(location, self.env.belief_info)
                trajectory_x = robot.trajectory_x.copy()
                trajectory_y = robot.trajectory_y.copy()
                trajectory_x.append(robot_location[0])
                trajectory_y.append(robot_location[1])
                # Draw highlighted trajectory (thicker, brighter)
                plt.plot((np.array(trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
                         (np.array(trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size, c,
                         linewidth=3.0, alpha=1.0, zorder=3, linestyle='--',
                         label=f'Detected: {color_name[plot_id]}')
                # Add a circle marker at current position
                plt.plot(location[0], location[1], 'o', color=c, markersize=8,
                         markeredgewidth=2, markeredgecolor='white', zorder=4)

        global_frontiers = get_frontier_in_map(self.env.belief_info)
        if len(global_frontiers) != 0:
            frontiers_cell = get_cell_position_from_coords(np.array(list(global_frontiers)), self.env.belief_info) #shape is (2,)
            if len(global_frontiers) == 1:
                frontiers_cell = frontiers_cell.reshape(1,2)
            plt.scatter(frontiers_cell[:, 0], frontiers_cell[:, 1], s=1, c='r')       

        # Top row - Panel 2: Global belief map with FOV cones
        plt.subplot(2, n_cols, 2)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.title('FOV & Detections', fontsize=10, fontweight='bold')

        # First pass: Draw all trajectories (thinner, semi-transparent)
        for i, (robot, location, heading) in enumerate(zip(self.robot_list, robot_locations, robot_headings)):
            plot_id = robot.id % 4
            c = color_list[plot_id]
            robot_location = get_coords_from_cell_position(location, self.env.belief_info)
            trajectory_x = robot.trajectory_x.copy()
            trajectory_y = robot.trajectory_y.copy()
            trajectory_x.append(robot_location[0])
            trajectory_y.append(robot_location[1])
            plt.plot((np.array(trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
                     (np.array(trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size, c,
                     linewidth=1.2, alpha=0.4, zorder=1)

        # Second pass: Draw FOV cones, arrows, and highlighted trajectories
        for i, (robot, location, heading) in enumerate(zip(self.robot_list, robot_locations, robot_headings)):
            plot_id = robot.id % 4
            c = color_list[plot_id]
            dx, dy = self.heading_to_vector(heading, length=sensing_range)
            arrow = FancyArrowPatch((location[0], location[1]), (location[0] + dx/1.25, location[1] + dy/1.25),
                                    mutation_scale=10,
                                    color=c,
                                    arrowstyle='-|>')
            plt.gca().add_artist(arrow)

            # Draw cone representing field of vision
            cone = Wedge(center=(location[0], location[1]), r=SENSOR_RANGE / CELL_SIZE, theta1=(heading-self.fov/2),
                         theta2=(heading+self.fov/2), color=c, alpha=0.3, zorder=10)
            plt.gca().add_artist(cone)

            # Highlight trajectory if detected by any other robot
            is_detected = False
            for detector_id, detected_list in fov_detections.items():
                if robot.id in detected_list:
                    is_detected = True
                    break

            if is_detected:
                robot_location = get_coords_from_cell_position(location, self.env.belief_info)
                trajectory_x = robot.trajectory_x.copy()
                trajectory_y = robot.trajectory_y.copy()
                trajectory_x.append(robot_location[0])
                trajectory_y.append(robot_location[1])
                # Draw highlighted trajectory (thicker, brighter)
                plt.plot((np.array(trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
                         (np.array(trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size, c,
                         linewidth=3.0, alpha=1.0, zorder=3, linestyle='--')
                # Add a circle marker at current position
                plt.plot(location[0], location[1], 'o', color=c, markersize=8,
                         markeredgewidth=2, markeredgecolor='white', zorder=15)

        # Draw detection connections (dashed lines from detector to detected robot)
        for detector_id, detected_list in fov_detections.items():
            for detected_id in detected_list:
                detector_loc = robot_locations[detector_id]
                detected_loc = robot_locations[detected_id]
                plt.plot([detector_loc[0], detected_loc[0]], [detector_loc[1], detected_loc[1]],
                         'w--', linewidth=1.5, alpha=0.6, zorder=11,
                         label='Detection' if detector_id == 0 and detected_id == detected_list[0] else '')

        # Plot frontiers
        if len(global_frontiers) != 0:
            plt.scatter(frontiers_cell[:, 0], frontiers_cell[:, 1], s=3, c='r')

        plt.axis('off')

        # Bottom row: Individual agent local views
        local_map_size = int(UPDATING_MAP_SIZE / CELL_SIZE)

        for robot in self.robot_list:
            plt.subplot(2, n_cols, n_cols + robot.id + 1)

            robot_location = get_coords_from_cell_position(robot_locations[robot.id], self.env.belief_info)
            plot_id = robot.id % 4
            c = color_list[plot_id]

            # Extract local map centered on robot
            center_cell = robot_locations[robot.id]
            half_size = local_map_size // 2

            row_start = max(0, int(center_cell[1] - half_size))
            row_end = min(self.env.robot_belief.shape[0], int(center_cell[1] + half_size))
            col_start = max(0, int(center_cell[0] - half_size))
            col_end = min(self.env.robot_belief.shape[1], int(center_cell[0] + half_size))

            local_map = self.env.robot_belief[row_start:row_end, col_start:col_end]

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
                mutation_scale=10,
                color=c,
                arrowstyle='-|>',
                linewidth=2
            )
            plt.gca().add_artist(arrow)

            # Draw FOV cone
            cone = Wedge(
                center=(robot_local_x, robot_local_y),
                r=SENSOR_RANGE / CELL_SIZE,
                theta1=(robot_headings[robot.id] - self.fov/2),
                theta2=(robot_headings[robot.id] + self.fov/2),
                color=c,
                alpha=0.3,
                zorder=10
            )
            plt.gca().add_artist(cone)

            # Draw other robots if they are in this local view
            for other_robot in self.robot_list:
                if other_robot.id == robot.id:
                    continue

                other_location = robot_locations[other_robot.id]
                other_local_x = other_location[0] - col_start
                other_local_y = other_location[1] - row_start

                # Check if other robot is within local view bounds
                if 0 <= other_local_x < local_map.shape[1] and 0 <= other_local_y < local_map.shape[0]:
                    other_plot_id = other_robot.id % 4
                    other_c = color_list[other_plot_id]

                    # Check if this other robot is detected by current robot
                    is_detected = other_robot.id in fov_detections.get(robot.id, [])

                    if is_detected:
                        # Highlight detected robots
                        plt.plot(other_local_x, other_local_y, 'o',
                                color=other_c, markersize=10,
                                markeredgewidth=3, markeredgecolor='yellow', zorder=15)
                        # Draw detection line
                        plt.plot([robot_local_x, other_local_x], [robot_local_y, other_local_y],
                                'y--', linewidth=2, alpha=0.8, zorder=12)
                    else:
                        # Draw non-detected robots with less emphasis
                        plt.plot(other_local_x, other_local_y, 'o',
                                color=other_c, markersize=6, alpha=0.5, zorder=5)

            # Draw local frontiers
            if robot.frontier:
                local_frontiers = []
                for frontier_coords in robot.frontier:
                    frontier_cell = get_cell_position_from_coords(np.array(frontier_coords), self.env.belief_info)
                    frontier_local_x = frontier_cell[0] - col_start
                    frontier_local_y = frontier_cell[1] - row_start
                    if 0 <= frontier_local_x < local_map.shape[1] and 0 <= frontier_local_y < local_map.shape[0]:
                        local_frontiers.append([frontier_local_x, frontier_local_y])

                if local_frontiers:
                    local_frontiers = np.array(local_frontiers)
                    plt.scatter(local_frontiers[:, 0], local_frontiers[:, 1], s=2, c='r', zorder=8)

            # Title for each agent's view
            detected_names = [color_name[did % 4] for did in fov_detections.get(robot.id, [])]
            detected_text = f"Detects: {', '.join(detected_names)}" if detected_names else "No detections"
            plt.title(f'{color_name[plot_id]} Agent Local View\n{detected_text}',
                     fontsize=9, fontweight='bold', color=c)

        # Build detection summary
        detection_summary = []
        for robot_id, detected_list in fov_detections.items():
            if len(detected_list) > 0:
                detected_colors = [color_name[did % 4] for did in detected_list]
                detection_summary.append(f"{color_name[robot_id % 4]} detects: {', '.join(detected_colors)}")

        detection_text = ' | '.join(detection_summary) if detection_summary else 'No detections'

        robot_headings_text = [f"{color_name[robot.id%4]}- {robot.heading:.0f}°" for robot in self.robot_list]
        plt.suptitle('Explored: {:.4g}  Distance: {:.4g}\nHeadings: {}\nFOV Detections: {}'.format(
            self.env.explored_rate,
            max([robot.travel_dist for robot in self.robot_list]),
            ', '.join(robot_headings_text),
            detection_text
        ), fontweight='bold', fontsize=10, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('{}/{}_{}_{}_{}_{}_samples.png'.format(gifs_path, self.global_step, step, self.n_agents, self.fov, self.sensor_range), dpi=150)
        plt.close()
        frame = '{}/{}_{}_{}_{}_{}_samples.png'.format(gifs_path, self.global_step, step, self.n_agents, self.fov, self.sensor_range)
        self.env.frame_files.append(frame)

    def correct_heading(self, heading):
        heading = abs(((heading + 90) % 360) - 360)
        return heading

if __name__ == '__main__':
    import torch
    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, use_trajectory=USE_TRAJECTORY)
    if LOAD_MODEL:
        checkpoint = torch.load(load_path + '/checkpoint.pth', map_location='cpu')
        policy_net.load_state_dict(checkpoint['policy_model'])
        print('Policy loaded!')
    worker = TestWorker(0, policy_net, 188, 4, 120, 10, 'cpu', True)
    worker.run_episode()
