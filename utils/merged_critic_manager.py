import numpy as np
import torch
import torch.nn.functional as F

from utils.node_manager import NodeManager
from utils.utils import *
from parameter import *


class MergedBeliefCriticManager:
    def __init__(self, fov, sensor_range, device='cpu', plot=False):
        self.fov = fov
        self.sensor_range = sensor_range
        self.device = device
        self.plot = plot
        self.num_angles_bin = NUM_ANGLES_BIN
        self.num_heading_candidates = NUM_HEADING_CANDIDATES
        self.cell_size = CELL_SIZE
        self.updating_map_size = UPDATING_MAP_SIZE
        self.node_manager = NodeManager(fov, sensor_range, plot=plot)
        self.map_info = None

    def update_graph(self, map_info, robot_locations):
        self.map_info = map_info
        for robot_location in robot_locations:
            updating_map_info = self.get_updating_map(robot_location)
            frontiers = get_frontier_in_map(updating_map_info)
            self.node_manager.update_graph(
                robot_location,
                frontiers,
                updating_map_info,
                self.map_info,
            )

    def get_updating_map(self, location):
        updating_map_origin_x = (location[0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[1] - self.updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = self.map_info.map_origin_x + self.cell_size * (self.map_info.map.shape[1] - 1)
        max_y = self.map_info.map_origin_y + self.cell_size * (self.map_info.map.shape[0] - 1)

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
            updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1] + 1,
            updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0] + 1,
        ]
        return MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

    def _get_all_node_coords(self):
        all_node_coords = [node.data.coords for node in self.node_manager.nodes_dict.__iter__()]
        return np.array(all_node_coords).reshape(-1, 2)

    def get_index_lookup(self):
        all_node_coords = self._get_all_node_coords()
        return {
            (float(coords[0]), float(coords[1])): index
            for index, coords in enumerate(all_node_coords)
        }

    def _build_merged_graph_state(self, robot_location):
        all_node_coords = self._get_all_node_coords()
        utility = []
        frontiers_distribution = []
        highest_utility_angle = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j

        for i, coords in enumerate(all_node_coords):
            node = self.node_manager.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            frontiers_distribution.append(node.frontiers_distribution)
            highest_utility_angle.append(node.highest_utility_angle)

            for neighbor in node.neighbor_list:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                if index.size > 0:
                    adjacent_matrix[i, index[0][0]] = 0

        utility = np.array(utility)
        frontiers_distribution = np.array(frontiers_distribution)
        highest_utility_angle = np.array(highest_utility_angle)

        indices = np.argwhere(utility > 0).reshape(-1)
        utility_node_coords = all_node_coords[indices]
        dist_dict, _ = self.node_manager.Dijkstra(robot_location)
        nearest_utility_coords = robot_location
        nearest_dist = 1e8
        for coords in utility_node_coords:
            dist = dist_dict[(coords[0], coords[1])]
            if 0 < dist < nearest_dist:
                nearest_dist = dist
                nearest_utility_coords = coords

        path_coords, _ = self.node_manager.a_star(robot_location, nearest_utility_coords)
        guidepost = np.zeros_like(utility)
        for coords in path_coords:
            index = np.argwhere(
                all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j
            )[0]
            guidepost[index] = 1

        robot_in_graph = self.node_manager.nodes_dict.nearest_neighbors(robot_location.tolist(), 1)[0].data.coords
        current_index = np.argwhere(
            node_coords_to_check == robot_in_graph[0] + robot_in_graph[1] * 1j
        )[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)

        occupancy = np.zeros((n_nodes, 1))
        occupancy[current_index] = -1

        return (
            all_node_coords,
            utility,
            guidepost,
            occupancy,
            adjacent_matrix,
            current_index,
            neighbor_indices,
            highest_utility_angle,
            frontiers_distribution,
            path_coords,
        )

    def _compute_best_heading(self, node_coords, frontier_distribution, neighbor_nodes, path_coords):
        neighbor_best_headings = []
        neighbor_nodes = list(neighbor_nodes[0])
        for i, neighbor in enumerate(neighbor_nodes):
            node_index = neighbor.item()
            heading_candidates = torch.zeros(self.num_heading_candidates, self.num_angles_bin)
            if (node_index != 0) or (i == 0 and node_index == 0):
                coords = node_coords[node_index]
                node_data = self.node_manager.nodes_dict.find((coords[0], coords[1])).data
                if node_data.utility > 0:
                    node_frontier_distribution = frontier_distribution[node_index]
                    half_fov_size = int((self.fov / 360) * self.num_angles_bin / 2)
                    window = np.concatenate((
                        node_frontier_distribution[-half_fov_size:],
                        node_frontier_distribution,
                        node_frontier_distribution[:half_fov_size],
                    ))
                    indices = np.arange(len(node_frontier_distribution)) + half_fov_size
                    sum_vector = np.sum(
                        np.take(
                            window,
                            indices.reshape(-1, 1) + np.arange(-half_fov_size, half_fov_size + 1),
                        ),
                        axis=1,
                    )
                    top_n_indices = np.argsort(-sum_vector)[:self.num_heading_candidates]
                    for offset in range(-half_fov_size, half_fov_size + 1):
                        indices = (top_n_indices + offset) % self.num_angles_bin
                        heading_candidates += F.one_hot(
                            torch.tensor(indices), num_classes=self.num_angles_bin
                        ).float()
                else:
                    top_n_indices = np.zeros(self.num_heading_candidates)
                    if len(path_coords) > 1:
                        next_coords = path_coords[1]
                        angle = np.degrees(np.arctan2(
                            next_coords[1] - coords[1],
                            next_coords[0] - coords[0],
                        ) % (2 * np.pi))
                        new_index = int(angle / 360 * self.num_angles_bin) % self.num_angles_bin
                        new_indices = [
                            (new_index + j - self.num_heading_candidates // 2) % self.num_angles_bin
                            for j in range(self.num_heading_candidates)
                        ]
                        for j in range(self.num_heading_candidates):
                            heading_candidates[j][
                                int(new_indices[j] - self.fov / 2):int(new_indices[j] + self.fov / 2)
                            ] = 1
                            top_n_indices[j] = new_indices[j]
                    else:
                        neighbor_list = node_data.neighbor_list[1:]
                        previous_index = 0
                        for j in range(self.num_heading_candidates):
                            if j < len(neighbor_list):
                                neighbor_coords = neighbor_list[j]
                                angle = np.degrees(np.arctan2(
                                    neighbor_coords[1] - coords[1],
                                    neighbor_coords[0] - coords[0],
                                ) % (2 * np.pi))
                                new_index = int(angle / 360 * self.num_angles_bin) % self.num_angles_bin
                                heading_candidates[j][
                                    int(new_index - self.fov / 2):int(new_index + self.fov / 2)
                                ] = 1
                                previous_index = new_index
                                top_n_indices[j] = new_index
                            else:
                                heading_candidates[j][previous_index + 1] = 1
                                top_n_indices[j] = previous_index + 1
                neighbor_best_headings.append(heading_candidates)
            else:
                neighbor_best_headings.append(heading_candidates)
        return torch.stack(neighbor_best_headings).unsqueeze(0).to(self.device)

    def get_critic_observation(self, robot_location, agent_node_manager, agent_map_info, pad=True):
        (
            node_coords,
            utility,
            guidepost,
            occupancy,
            merged_edge_mask,
            current_index,
            merged_current_edge,
            highest_utility_angles,
            merged_frontier_distribution,
            path_coords,
        ) = self._build_merged_graph_state(robot_location)

        n_node = node_coords.shape[0]
        current_node_coords = node_coords[current_index]

        all_node_coords = np.concatenate((
            node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
            node_coords[:, 1].reshape(-1, 1) - current_node_coords[1],
        ), axis=-1) / UPDATING_MAP_SIZE / 2

        node_utility = utility.reshape(-1, 1) / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        node_guidepost = guidepost.reshape(-1, 1)
        node_occupancy = occupancy.reshape(-1, 1)
        node_highest_utility_angles = highest_utility_angles.reshape(-1, 1) / 360

        # Critic utility/frontier/neighbor structure comes strictly from the merged team map.
        # Only these memory-like channels remain agent-specific.
        node_heading_visited = []
        node_visited_by_others = []
        node_other_agents_explored = []
        node_cells = get_cell_position_from_coords(node_coords, self.map_info).reshape(-1, 2)

        for coords, cell in zip(node_coords, node_cells):
            local_node = agent_node_manager.nodes_dict.find((coords[0], coords[1]))
            if local_node is not None:
                node_heading_visited.append(local_node.data.heading_visited)
                node_visited_by_others.append(local_node.data.visited_by_others)
            else:
                node_heading_visited.append(np.zeros(self.num_angles_bin))
                node_visited_by_others.append(0.0)

            merged_value = self.map_info.map[cell[1], cell[0]]
            agent_value = agent_map_info.map[cell[1], cell[0]]
            node_other_agents_explored.append(float(merged_value != UNKNOWN and agent_value == UNKNOWN))

        node_heading_visited = np.array(node_heading_visited).reshape(-1, self.num_angles_bin)
        node_visited_by_others = np.array(node_visited_by_others).reshape(-1, 1)
        node_other_agents_explored = np.array(node_other_agents_explored).reshape(-1, 1)
        node_frontier_distribution = merged_frontier_distribution.reshape(-1, self.num_angles_bin)
        node_frontier_distribution = node_frontier_distribution / (
            (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE) / self.num_angles_bin
        )

        node_inputs = np.concatenate((
            all_node_coords,
            node_utility,
            node_guidepost,
            node_occupancy,
            node_highest_utility_angles,
            node_visited_by_others,
            node_other_agents_explored,
        ), axis=1)

        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
        node_frontier_distribution = torch.FloatTensor(node_frontier_distribution).unsqueeze(0).to(self.device)
        node_heading_visited = torch.FloatTensor(node_heading_visited).unsqueeze(0).to(self.device)
        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)

        if pad:
            padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
            node_inputs = padding(node_inputs)
            node_frontier_distribution = padding(node_frontier_distribution)
            node_heading_visited = padding(node_heading_visited)

            node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(self.device)
            node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        current_index_tensor = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)
        edge_mask_tensor = torch.tensor(merged_edge_mask).unsqueeze(0).to(self.device)
        if pad:
            padding = torch.nn.ConstantPad2d(
                (0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1
            )
            edge_mask_tensor = padding(edge_mask_tensor)

        current_in_edge = np.argwhere(merged_current_edge == current_index)[0][0]
        current_edge_tensor = torch.tensor(merged_current_edge).unsqueeze(0)
        k_size = current_edge_tensor.size()[-1]
        if pad:
            padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
            current_edge_tensor = padding(current_edge_tensor)
        current_edge_tensor = current_edge_tensor.unsqueeze(-1)

        critic_neighbor_best_headings = self._compute_best_heading(
            node_coords,
            merged_frontier_distribution,
            current_edge_tensor,
            path_coords,
        )

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        if pad:
            padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
            edge_padding_mask = padding(edge_padding_mask)

        return [
            node_inputs,
            node_padding_mask,
            edge_mask_tensor,
            current_index_tensor,
            current_edge_tensor,
            edge_padding_mask,
            node_frontier_distribution,
            node_heading_visited,
            critic_neighbor_best_headings,
        ]
