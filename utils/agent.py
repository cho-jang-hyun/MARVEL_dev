"""
Agent class for multi-robot exploration using a policy network.

This class manages an individual agent's state, movement and mapping in a multi-robot exploration environment. It handles key functionalities 
such as:
- Tracking agent location, heading, and travel distance
- Updating map and frontier information
- Managing graph-based exploration
- Generating observations for policy network
- Selecting next waypoints
- Saving episode data for learning

Attributes:
    id (int): Unique identifier for the agent
    policy_net (torch.nn.Module): Neural network for action selection
    location (np.ndarray): Current agent location
    heading (float): Current agent heading in degrees
    sensor_range (float): Maximum sensing range of the agent
    node_manager (NodeManager): Manages graph nodes 
"""
import copy
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from skimage.draw import polygon as sk_polygon

from utils.utils import *
from parameter import *

class Agent:
    def __init__(self, id, policy_net, fov, heading, sensor_range, node_manager, ground_truth_node_manager, device='cpu', plot=False, base_location=None):
        self.id = id
        self.device = device
        self.plot = plot
        self.policy_net = policy_net
        self.fov = fov
        self.num_prev_headings = 3
        self.sensor_range = sensor_range
        self.num_angles_bin = NUM_ANGLES_BIN
        self.num_heading_candidates = NUM_HEADING_CANDIDATES
        self.base_location = base_location

        # location and global map
        self.location = None
        self.map_info = None

        # Motion parameters
        self.velocity = VELOCITY
        self.yaw_rate = YAW_RATE

        # map related parameters
        self.cell_size = CELL_SIZE
        self.node_resolution = NODE_RESOLUTION
        self.updating_map_size = UPDATING_MAP_SIZE

        # map and updating map
        self.map_info = None
        self.updating_map_info = None

        # frontiers
        self.frontier = set()

        # node managers
        self.node_manager = node_manager
        self.ground_truth_node_manager = ground_truth_node_manager

        # graph
        self.node_coords, self.utility, self.guidepost, self.occupancy = None, None, None, None
        self.current_index, self.adjacent_matrix, self.neighbor_indices = None, None, None

        self.highest_utility_angles, self.frontier_distribution, self.heading_visited, self.visited_by_others = None, None, None, None
        self.path_coords = None

        self.travel_dist = 0

          # Heading info
        angle = 0 if heading == 360 else heading
        self.heading = angle

        self.episode_buffer = []
        for i in range(NUM_EPISODE_BUFFER):
            self.episode_buffer.append([])

        if self.plot:
            self.trajectory_x = []
            self.trajectory_y = []


    def update_map(self, map_info):
        # no need in training because of shallow copy
        self.map_info = map_info

    def update_updating_map(self, location):
        self.updating_map_info = self.get_updating_map(location)

    def update_location(self, location):
        if self.location is None:
            dist = 0
        else:
            dist = np.linalg.norm(self.location - location)
        self.travel_dist += dist

        self.location = location

        node = self.node_manager.nodes_dict.find(location.tolist())
        if self.node_manager.nodes_dict.__len__() == 0:
            pass
        else:
            node.data.set_visited(self.heading)
            
        if self.plot:
            self.trajectory_x.append(location[0])
            self.trajectory_y.append(location[1])

    def update_frontiers(self):
        self.frontier = get_frontier_in_map(self.updating_map_info)

    def update_heading(self, heading):
        # Update heading data
        self.heading = heading
        
    def get_updating_map(self, location):
        updating_map_origin_x = (location[0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[1] - self.updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = (self.map_info.map_origin_x + self.cell_size * (self.map_info.map.shape[1] - 1))
        max_y = (self.map_info.map_origin_y + self.cell_size * (self.map_info.map.shape[0] - 1))

        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
        updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
        updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, self.map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, self.map_info)

        updating_map = self.map_info.map[
                    updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1]+1,
                    updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0]+1]

        updating_map_info = MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

        return updating_map_info

    def update_graph(self, map_info, location):
        self.update_map(map_info)
        self.update_location(location)
        self.update_updating_map(self.location)
        self.update_frontiers()
        self.node_manager.update_graph(self.location,
                                       self.frontier,
                                       self.updating_map_info,
                                       self.map_info)

    def update_planning_state(self):
        self.node_coords, self.utility, self.guidepost, self.occupancy, self.adjacent_matrix, self.current_index, self.neighbor_indices, self.highest_utility_angles, self.frontier_distribution, self.heading_visited, self.visited_by_others, self.path_coords = \
            self.node_manager.get_all_node_graph(self.location)

    def get_observation(self, pad=True, robot_locations=None, trajectory_buffer=None, remaining_budget=None):
        node_coords = self.node_coords
        node_utility = self.utility.reshape(-1, 1)
        node_guidepost = self.guidepost.reshape(-1, 1)
        node_occupancy = self.occupancy.reshape(-1, 1)
        node_highest_utility_angles = self.highest_utility_angles.reshape(-1, 1)
        node_frontier_distribution = self.frontier_distribution.reshape(-1, self.num_angles_bin)
        node_heading_visited = self.heading_visited.reshape(-1, self.num_angles_bin)
        node_visited_by_others = self.visited_by_others.reshape(-1, 1)
        current_index = self.current_index
        edge_mask = self.adjacent_matrix
        current_edge = self.neighbor_indices
        n_node = node_coords.shape[0]

        current_node_coords = node_coords[self.current_index]
        all_node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                             node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                           axis=-1) / UPDATING_MAP_SIZE / 2
        node_utility = node_utility / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        node_highest_utility_angles = node_highest_utility_angles / 360
        node_frontier_distribution = node_frontier_distribution / ((2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE) / self.num_angles_bin)
        # visited_by_others is already 0 or 1, no normalization needed
        # budget and RTB features
        dist_to_base = np.linalg.norm(node_coords - self.base_location, axis=1).reshape(-1, 1) if self.base_location is not None else np.zeros((n_node, 1))
        # Normalize distance and budget
        node_dist_to_base = dist_to_base / (MAX_EPISODE_STEP * VELOCITY)
        node_remaining_budget = (np.ones((n_node, 1)) * remaining_budget / BUDGET) if remaining_budget is not None else np.zeros((n_node, 1))

        node_inputs = np.concatenate((all_node_coords, node_utility, node_guidepost, node_occupancy, node_highest_utility_angles, node_visited_by_others, node_remaining_budget, node_dist_to_base), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
        all_node_frontier_distribution = torch.Tensor(node_frontier_distribution).unsqueeze(0).to(self.device)
        node_heading_visited = torch.Tensor(node_heading_visited).unsqueeze(0).to(self.device)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)

        if pad:
            padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
            node_inputs = padding(node_inputs)
            all_node_frontier_distribution = padding(all_node_frontier_distribution)
            node_heading_visited = padding(node_heading_visited)

            node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(
                self.device)
            node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)

        if pad:
            padding = torch.nn.ConstantPad2d(
                (0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
            edge_mask = padding(edge_mask)

        current_in_edge = np.argwhere(current_edge == self.current_index)[0][0]
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        if pad:
            padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
            current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)
     
        node_neighbor_best_headings, self.neighbor_best_indices = self.compute_best_heading(node_coords, node_frontier_distribution, current_edge)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1

        if pad:
            padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
            edge_padding_mask = padding(edge_padding_mask)

        # Process trajectory information if available
        detected_trajectories = None
        trajectory_mask = None
        trajectory_node_indices = None
        if robot_locations is not None and trajectory_buffer is not None:
            detected_trajectories, trajectory_mask, trajectory_node_indices = self._get_detected_trajectories(
                robot_locations, trajectory_buffer
            )

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask,
                all_node_frontier_distribution, node_heading_visited, node_neighbor_best_headings,
                detected_trajectories, trajectory_mask, trajectory_node_indices]

    def select_next_waypoint(self, observation, greedy = False):
        _, _, _, _, current_edge, _, _, _, _, _, _, _ = observation
        with torch.no_grad():
            logp = self.policy_net(*observation[:9], detected_trajectories=observation[9],
                                   trajectory_mask=observation[10], trajectory_node_indices=observation[11])

        if greedy:
            action_index = torch.argmax(logp, 1).long()
        else:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)

        waypoint_index = action_index.item() // self.num_heading_candidates
        next_node_index = current_edge[0, waypoint_index, 0].item()
        heading_index = self.neighbor_best_indices[waypoint_index][action_index.item() % self.num_heading_candidates]
        next_position = self.node_coords[next_node_index]

        return next_position, next_node_index, action_index, heading_index
    
    def compute_best_heading(self, node_coords, frontier_distribution, neighbor_nodes):
        neighbor_best_headings = []
        neighbor_best_indices = []
        neighbor_nodes = list(neighbor_nodes[0])
        for i, neighbor in enumerate(neighbor_nodes):
            node_index = neighbor.item()
            heading_candidates = torch.zeros(self.num_heading_candidates, self.num_angles_bin)
            if (node_index != 0) or (i == 0 and node_index == 0):
                coords = node_coords[node_index]
                node_data = self.node_manager.nodes_dict.find((coords[0], coords[1])).data
                if node_data.utility > 0:
                    node_frontier_distribution = frontier_distribution[node_index]
                    half_fov_size = int((self.fov / 360)*self.num_angles_bin/2)
                    window = np.concatenate((node_frontier_distribution[-half_fov_size:], node_frontier_distribution, node_frontier_distribution[:half_fov_size]))
                    indices = np.arange(len(node_frontier_distribution)) + half_fov_size
                    sum_vector = np.sum(np.take(window, indices.reshape(-1, 1) + np.arange(-half_fov_size, half_fov_size + 1)), axis=1)
                    top_n_indices = np.argsort(-sum_vector)[:self.num_heading_candidates]
                    for i in range(-half_fov_size, half_fov_size+1):
                        indices = (top_n_indices + i) % self.num_angles_bin
                        heading_candidates += F.one_hot(torch.tensor(indices), num_classes=self.num_angles_bin).float()
                else:
                    top_n_indices = np.zeros(3)
                    # Face the robot towards the A* path within 1 bin variance
                    if len(self.path_coords) > 1:
                        next_coords = self.path_coords[1]
                        angle = np.degrees(np.arctan2(next_coords[1] - coords[1],
                                                next_coords[0] - coords[0]) % (2 * np.pi))
                        new_index = int(angle / 360 * self.num_angles_bin) % self.num_angles_bin
                        new_indices = [(new_index + i - self.num_heading_candidates // 2) % self.num_angles_bin for i in range(self.num_heading_candidates)]
                        for l in range(self.num_heading_candidates):
                            heading_candidates[l][int(new_indices[l] - self.fov/2):int(new_indices[l] + self.fov/2)] = 1
                            top_n_indices[l] = new_indices[l]
                    else:
                        for l in range(self.num_heading_candidates):
                            # Make the robot face the direction of neighbor nodes
                            neighbor_list = node_data.neighbor_list[1:]     # First node is self
                            for l in range(self.num_heading_candidates):
                                previous_index = 0
                                if l < len(neighbor_list):
                                    neighbor_coords = neighbor_list[l]                                     # First node is self
                                    angle = np.degrees(np.arctan2(neighbor_coords[1] - coords[1], 
                                                            neighbor_coords[0] - coords[0]) % (2 * np.pi))
                                    new_index = int(angle / 360 * self.num_angles_bin) % self.num_angles_bin
                                    heading_candidates[l][int(new_index-self.fov/2):int(new_index+self.fov/2)] = 1
                                    previous_index = new_index
                                    top_n_indices[l] = new_index
                                else:
                                    heading_candidates[l][previous_index+1] = 1
                                    top_n_indices[l] = previous_index+1
                neighbor_best_headings.append(heading_candidates)
                neighbor_best_indices.append(top_n_indices)
            else:
                neighbor_best_headings.append(heading_candidates)
                neighbor_best_indices.append(np.zeros(3))
        neighbor_best_headings = torch.stack(neighbor_best_headings).unsqueeze(0).to(self.device)
        return neighbor_best_headings, neighbor_best_indices
    
    def check_coords_in_path(self, coords):
        if coords in self.path_coords:
            index = self.path_coords.index(coords)
            next_coord = self.path_coords[index + 1] if index + 1 < len(self.path_coords) else None
            return next_coord
        return  None
    
    def heading_to_vector(self, heading, length=25):
        # Convert heading to vector
        if isinstance(heading, (list, np.ndarray)):
            heading = heading[0]
        heading_rad = np.radians(heading)
        return np.cos(heading_rad) * length, np.sin(heading_rad) * length
    
    def create_sensing_mask(self, location, heading, mask):

        location_cell = get_cell_position_from_coords(location, self.map_info)
        # Create a Point for the robot's location
        robot_point = Point(location_cell)

        # Calculate the angles for the sector
        start_angle = (heading - self.fov / 2 + 360) % 360
        end_angle = (heading + self.fov / 2) % 360

        # Create points for the sector
        sector_points = [robot_point]
        if start_angle <= end_angle:
            angle_range = np.linspace(start_angle, end_angle, 20)
        else:
            angle_range = np.concatenate([np.linspace(start_angle, 360, 10), np.linspace(0, end_angle, 10)])
        for angle in angle_range:
            x = location_cell[0] + self.sensor_range/CELL_SIZE * np.cos(np.radians(angle))
            y = location_cell[1] + self.sensor_range/CELL_SIZE * np.sin(np.radians(angle))
            sector_points.append(Point(x, y))
        sector_points.append(robot_point)
        # Create the sector polygon
        sector = Polygon(sector_points)

        x_coords, y_coords = sector.exterior.xy
        y_coords = np.rint(y_coords).astype(int)
        x_coords = np.rint(x_coords).astype(int)
        rr, cc = sk_polygon(
                [int(round(y)) for y in y_coords],
                [int(round(x)) for x in x_coords],
                shape=mask.shape
            )

        free_connected_map = get_free_and_connected_map(location, self.map_info)

        mask[rr, cc] = (free_connected_map[rr, cc] == free_connected_map[location_cell[1], location_cell[0]])

        return mask

    def get_robots_in_fov(self, robot_locations, observer_location=None, observer_heading=None):
        """
        Detect other robots within the agent's field of view (FOV).

        Args:
            robot_locations (np.ndarray): Array of all robot locations, shape (n_agents, 2)
            observer_location (np.ndarray, optional): Override observer location.
            observer_heading (float, optional): Override observer heading.

        Returns:
            List[int]: List of agent IDs that are within FOV and sensor range
        """
        if observer_location is None:
            observer_location = self.location
        if observer_heading is None:
            observer_heading = self.heading

        detected_robots = []

        for robot_id, robot_location in enumerate(robot_locations):
            # Skip self
            if robot_id == self.id:
                continue

            # Calculate distance
            distance = np.linalg.norm(robot_location - observer_location)

            # Check if within sensor range
            if distance > self.sensor_range:
                continue

            # Calculate angle to the other robot
            delta = robot_location - observer_location
            angle_to_robot = np.degrees(np.arctan2(delta[1], delta[0])) % 360

            # Calculate angle difference considering FOV
            angle_diff = (angle_to_robot - observer_heading + 180) % 360 - 180

            # Check if within FOV
            if np.abs(angle_diff) <= self.fov / 2:
                # Line-of-sight checks
                if not check_collision(observer_location, robot_location, self.map_info, allow_unknown=True):
                    detected_robots.append(robot_id)

        return detected_robots

    def _get_detected_trajectories(self, robot_locations, trajectory_buffer):
        """
        Extract trajectories of detected robots within FOV and convert to observation format.

        Args:
            robot_locations (np.ndarray): Array of all robot locations, shape (n_agents, 2)
            trajectory_buffer (dict): Dictionary mapping agent_id to deque of (x, y, heading, velocity)

        Returns:
            detected_trajectories: torch.Tensor of shape [1, MAX_DETECTED_AGENTS, seq_len, feature_dim]
            trajectory_mask: torch.Tensor of shape [1, MAX_DETECTED_AGENTS], True for padded agents
            trajectory_node_indices: torch.Tensor of shape [1, MAX_DETECTED_AGENTS], containing the latest node index
        """
        if not hasattr(self, 'unseen_teammates'):
            self.unseen_teammates = {}

        # Detect robots in FOV
        detected_robot_ids = self.get_robots_in_fov(robot_locations)

        # Update own tracking
        for robot_id in range(len(robot_locations)):
            if robot_id == self.id:
                continue

            if robot_id in detected_robot_ids:
                # Agent is in FOV, get current node
                current_step = trajectory_buffer[robot_id][-1]
                x, y, heading, velocity = current_step
                position = np.array([x, y])
                nearest_idx = -1
                if self.node_coords is not None and len(self.node_coords) > 0:
                    distances = np.linalg.norm(self.node_coords - position, axis=1)
                    if distances.min() < NODE_RESOLUTION:
                        nearest_idx = np.argmin(distances)

                self.unseen_teammates[robot_id] = {
                    "node": nearest_idx,
                    "steps": 0
                }
            else:
                if robot_id in self.unseen_teammates:
                    self.unseen_teammates[robot_id]["steps"] += 1
                    # Hard forget if unseen for too long (e.g. 100 steps)
                    if self.unseen_teammates[robot_id]["steps"] > 100:
                        del self.unseen_teammates[robot_id]

        max_agents = MAX_DETECTED_AGENTS
        
        unseen_steps = np.zeros(max_agents, dtype=np.float32)
        trajectory_node_indices = np.full(max_agents, -1, dtype=np.int64)
        mask = np.ones(max_agents, dtype=bool)
        
        tracked_ids = list(self.unseen_teammates.keys())
        for i, robot_id in enumerate(tracked_ids[:max_agents]):
            info = self.unseen_teammates[robot_id]
            if info["node"] != -1:
                unseen_steps[i] = info["steps"]
                trajectory_node_indices[i] = info["node"]
                mask[i] = False
                
        detected_trajectories = torch.FloatTensor(unseen_steps).unsqueeze(0).to(self.device)
        trajectory_mask = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        trajectory_node_indices_tensor = torch.LongTensor(trajectory_node_indices).unsqueeze(0).to(self.device)

        return detected_trajectories, trajectory_mask, trajectory_node_indices_tensor

    def mark_nodes_visited_by_others(self, robot_locations, trajectory_buffer):
        """
        Mark nodes as visited by other agents based on observed trajectories in FoV.

        This method:
        1. Detects other robots in FoV using get_robots_in_fov()
        2. For each detected robot, retrieves its trajectory history
        3. Identifies which nodes the detected robots have visited
        4. Marks those nodes with visited_by_others = 1

        Args:
            robot_locations (np.ndarray): Array of all robot locations, shape (n_agents, 2)
            trajectory_buffer (dict): Dictionary mapping agent_id to deque of (x, y, heading, velocity)

        Returns:
            None (modifies node_manager nodes in place)
        """
        if not hasattr(self, 'unseen_teammates'):
            return
            
        for robot_id, info in self.unseen_teammates.items():
            node_idx = info["node"]
            if node_idx != -1 and node_idx < len(self.node_coords):
                node_coord = self.node_coords[node_idx]
                node = self.node_manager.nodes_dict.find((node_coord[0], node_coord[1]))
                if node is not None:
                    node.data.visited_by_others = 1

    def calculate_overlap_reward(self, current_robot_location, all_robots_locations, robot_headings_list):
        ## Robot heading list in degrees
        current_sensing_mask = np.zeros_like(self.map_info.map)
        other_robot_sensing_mask = np.zeros_like(self.map_info.map)

        # Get robots in FoV (no communication - observation only)
        observer_heading = robot_headings_list[self.id]
        detected_robot_ids = self.get_robots_in_fov(
            all_robots_locations,
            observer_location=current_robot_location,
            observer_heading=observer_heading
        )

        for robot_id, (robot_location, robot_heading) in enumerate(zip(all_robots_locations, robot_headings_list)):
            if robot_id == self.id:
                current_sensing_mask = self.create_sensing_mask(current_robot_location, robot_heading, current_sensing_mask)
            elif robot_id in detected_robot_ids:
                # Only consider robots within FoV
                other_robot_sensing_mask = self.create_sensing_mask(robot_location, robot_heading, other_robot_sensing_mask)

        # Keep cell value of 1 only for cells that hold a value of 255 in self.global_map_info.map
        current_free_area_size = np.sum(current_sensing_mask)
        unique_sensing_mask = np.logical_and(current_sensing_mask == 1, other_robot_sensing_mask == 0).astype(int)
        # Compute the number of cells that have a value of 1 in current_sensing_mask and 0 in other_robot_sensing_mask
        current_free_area_not_scanned_size = np.sum(unique_sensing_mask)

        # Handle division by zero: if no free area in FoV, return 0 penalty
        if current_free_area_size == 0:
            overlap_penalty = 0.0
        else:
            # Calculate overlap ratio (how much of current sensing area overlaps with others)
            overlap_area_size = current_free_area_size - current_free_area_not_scanned_size
            overlap_penalty = overlap_area_size / current_free_area_size

        return overlap_penalty

    def _get_empty_trajectory_tensors(self):
        detected_trajectories = torch.zeros(
            (1, MAX_DETECTED_AGENTS),
            dtype=torch.float32,
            device=self.device,
        )
        trajectory_mask = torch.ones(
            (1, MAX_DETECTED_AGENTS),
            dtype=torch.bool,
            device=self.device,
        )
        trajectory_node_indices = torch.full(
            (1, MAX_DETECTED_AGENTS),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        return detected_trajectories, trajectory_mask, trajectory_node_indices
        
    def save_observation(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontier_distribution, heading_visited, neighbor_best_headings, detected_trajectories, trajectory_mask, trajectory_node_indices = observation
        if detected_trajectories is None or trajectory_mask is None or trajectory_node_indices is None:
            detected_trajectories, trajectory_mask, trajectory_node_indices = self._get_empty_trajectory_tensors()
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()
        self.episode_buffer[6] += frontier_distribution
        self.episode_buffer[7] += heading_visited
        self.episode_buffer[38] += neighbor_best_headings
        self.episode_buffer[40] += detected_trajectories
        self.episode_buffer[41] += trajectory_mask
        self.episode_buffer[42] += trajectory_node_indices

    def save_action(self, action_index):
        self.episode_buffer[8] += action_index.reshape(1, 1, 1)

    def save_reward(self, reward):
        self.episode_buffer[9] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)

    def save_done(self, done):
        self.episode_buffer[10] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_next_observations(self, observation, next_node_index_list):
        self.episode_buffer[11] = copy.deepcopy(self.episode_buffer[0])[1:]
        self.episode_buffer[12] = copy.deepcopy(self.episode_buffer[1])[1:]
        self.episode_buffer[13] = copy.deepcopy(self.episode_buffer[2])[1:]
        self.episode_buffer[14] = copy.deepcopy(self.episode_buffer[3])[1:]
        self.episode_buffer[15] = copy.deepcopy(self.episode_buffer[4])[1:]
        self.episode_buffer[16] = copy.deepcopy(self.episode_buffer[5])[1:]
        self.episode_buffer[17] = copy.deepcopy(self.episode_buffer[6])[1:]
        self.episode_buffer[18] = copy.deepcopy(self.episode_buffer[7])[1:]

        # Only process agent indices if they were saved (USE_COMMUNICATION=True)
        if len(self.episode_buffer[35]) > 0:
            self.episode_buffer[36] = copy.deepcopy(self.episode_buffer[35])[1:]

        self.episode_buffer[39] = copy.deepcopy(self.episode_buffer[38])[1:]
        self.episode_buffer[43] = copy.deepcopy(self.episode_buffer[40])[1:]
        self.episode_buffer[44] = copy.deepcopy(self.episode_buffer[41])[1:]
        self.episode_buffer[45] = copy.deepcopy(self.episode_buffer[42])[1:]

        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontier_distribution, heading_visited, neighbor_best_headings, detected_trajectories, trajectory_mask, trajectory_node_indices = observation
        if detected_trajectories is None or trajectory_mask is None or trajectory_node_indices is None:
            detected_trajectories, trajectory_mask, trajectory_node_indices = self._get_empty_trajectory_tensors()
        self.episode_buffer[11] += node_inputs
        self.episode_buffer[12] += node_padding_mask.bool()
        self.episode_buffer[13] += edge_mask.bool()
        self.episode_buffer[14] += current_index
        self.episode_buffer[15] += current_edge
        self.episode_buffer[16] += edge_padding_mask.bool()
        self.episode_buffer[17] += frontier_distribution
        self.episode_buffer[18] += heading_visited
        self.episode_buffer[39] += neighbor_best_headings
        self.episode_buffer[43] += detected_trajectories
        self.episode_buffer[44] += trajectory_mask
        self.episode_buffer[45] += trajectory_node_indices

        # Only update agent indices buffers if they were initialized
        if len(self.episode_buffer[35]) > 0:
            self.episode_buffer[36] += torch.tensor(next_node_index_list).reshape(1, -1, 1).to(self.device)
            self.episode_buffer[37] = copy.deepcopy(self.episode_buffer[36])[1:]
            self.episode_buffer[37] += copy.deepcopy(self.episode_buffer[36])[-1:]

    def save_ground_truth_observation(self, ground_truth_observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontier_distribution, heading_visited = ground_truth_observation
        self.episode_buffer[19] += node_inputs
        self.episode_buffer[20] += node_padding_mask.bool()
        self.episode_buffer[21] += edge_mask.bool()
        self.episode_buffer[22] += current_index
        self.episode_buffer[23] += current_edge
        self.episode_buffer[24] += edge_padding_mask.bool()
        self.episode_buffer[25] += frontier_distribution
        self.episode_buffer[26] += heading_visited

    def save_next_ground_truth_observations(self, ground_truth_observation):
        self.episode_buffer[27] = copy.deepcopy(self.episode_buffer[19])[1:]
        self.episode_buffer[28] = copy.deepcopy(self.episode_buffer[20])[1:]
        self.episode_buffer[29] = copy.deepcopy(self.episode_buffer[21])[1:]
        self.episode_buffer[30] = copy.deepcopy(self.episode_buffer[22])[1:]
        self.episode_buffer[31] = copy.deepcopy(self.episode_buffer[23])[1:]
        self.episode_buffer[32] = copy.deepcopy(self.episode_buffer[24])[1:]
        self.episode_buffer[33] = copy.deepcopy(self.episode_buffer[25])[1:]
        self.episode_buffer[34] = copy.deepcopy(self.episode_buffer[26])[1:]

        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontier_distribution, heading_visited = ground_truth_observation
        self.episode_buffer[27] += node_inputs
        self.episode_buffer[28] += node_padding_mask.bool()
        self.episode_buffer[29] += edge_mask.bool()
        self.episode_buffer[30] += current_index
        self.episode_buffer[31] += current_edge
        self.episode_buffer[32] += edge_padding_mask.bool()
        self.episode_buffer[33] += frontier_distribution
        self.episode_buffer[34] += heading_visited

    def save_all_indices(self, all_agent_indices):
        self.episode_buffer[35] += torch.tensor(all_agent_indices).reshape(1, -1, 1).to(self.device)
