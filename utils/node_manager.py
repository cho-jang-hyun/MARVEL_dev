"""
Manages nodes in a quadtree-based spatial graph for exploration and frontier tracking.

This class handles the creation, updating, and management of nodes representing 
spatial locations during exploration. It supports tracking observable frontiers, 
calculating node utilities, managing neighbor relationships, and performing 
pathfinding algorithms like Dijkstra and A* on the node graph.

Key functionalities:
- Add, remove, and update nodes in a quadtree data structure
- Calculate node utilities based on observable frontiers
- Manage neighbor node connections
- Perform graph-based pathfinding algorithms
"""
import heapq
import time
from collections import deque

import numpy as np
from utils.utils import *
from utils.runtime_config import *
import utils.quads as quads


class NodeManager:
    def __init__(self, fov, sensor_range, utility_range=None, plot=False):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.plot = plot
        self.fov = fov
        self.sensor_range = sensor_range
        if utility_range is None:
            self.utility_range = UTILITY_RANGE
        else:
            self.utility_range = utility_range
        self._dijkstra_cache = {}
        # Guidepost hysteresis: remember last frontier target to avoid flip-flopping
        # between equidistant clusters every step.
        self._last_guidepost_target = None
        self._GUIDEPOST_HYSTERESIS = 0.80  # new target must be ≥20% closer to switch
        self._astar_cache = {}
        self._bfs_cache = {}

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.nodes_dict.find(key)
        return exist

    def add_node_to_dict(self, coords, local_frontiers, updating_map_info):
        key = (coords[0], coords[1])
        node = Node(coords, local_frontiers, updating_map_info, self.fov, self.sensor_range, utility_range=self.utility_range)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def remove_node_from_dict(self, node):
        for neighbor_coords in node.neighbor_list[1:]:
            neighbor_node = self.nodes_dict.find(neighbor_coords)
            neighbor_node.data.neighbor_list.remove(node.coords.tolist())
        self.nodes_dict.remove(node.coords.tolist())

    def update_graph(self, robot_location, frontiers, updating_map_info, map_info,
                     skip_far_existing_updates=True, refresh_all_neighbors=False):
        self._dijkstra_cache.clear()
        self._astar_cache.clear()
        self._bfs_cache.clear()
        node_coords, _ = get_updating_node_coords(robot_location, updating_map_info)

        all_node_list = []
        for coords in node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                node = self.add_node_to_dict(coords, frontiers, updating_map_info)
            else:
                node = node.data
                too_far_to_refresh = (
                    skip_far_existing_updates
                    and np.linalg.norm(node.coords - robot_location) > 2 * self.sensor_range
                )
                if too_far_to_refresh:
                    pass
                else:
                    node.update_node_observable_frontiers(frontiers, updating_map_info, map_info)
            all_node_list.append(node)

        neighbor_map_info = map_info if refresh_all_neighbors else updating_map_info
        for node in all_node_list:
            should_refresh_neighbor = refresh_all_neighbors or (
                np.linalg.norm(node.coords - robot_location) < (self.sensor_range + NODE_RESOLUTION)
            )
            if node.need_update_neighbor and should_refresh_neighbor:
                node.update_neighbor_nodes(neighbor_map_info, self.nodes_dict)

    def get_all_node_graph(self, robot_location):
        all_node_coords = []
        for node in self.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        frontiers_distribution = []
        highest_utility_angle = []
        heading_visited = []
        visited_by_others = []
        visited_self = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        coord_to_index = {(c[0], c[1]): i for i, c in enumerate(all_node_coords)}
        for i, coords in enumerate(all_node_coords):
            node = self.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            frontiers_distribution.append(node.frontiers_distribution)
            heading_visited.append(node.heading_visited)
            highest_utility_angle.append(node.highest_utility_angle)
            visited_self.append(float(node.visited))

            # Decay visited_by_others over time but keep a small persistent memory once a teammate was seen.
            if node.visited_by_others > 0.0:
                node.visited_by_others = max(VISITED_BY_OTHERS_MIN, node.visited_by_others - VISITED_BY_OTHERS_DECAY)
            visited_by_others.append(node.visited_by_others)

            for neighbor in node.neighbor_list:
                index = coord_to_index.get((neighbor[0], neighbor[1]))
                if index is not None:
                    adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        frontiers_distribution = np.array(frontiers_distribution)
        highest_utility_angle = np.array(highest_utility_angle)
        heading_visited = np.array(heading_visited)
        visited_by_others = np.array(visited_by_others)
        visited_self = np.array(visited_self)

        indices = np.argwhere(utility > 0).reshape(-1)
        utility_node_coords = all_node_coords[indices]
        dist_dict, prev_dict = self.Dijkstra(robot_location)

        # Hysteresis: keep the previous target unless a new one is meaningfully
        # closer (by at least HYSTERESIS_FACTOR), preventing the guidepost from
        # flip-flopping between equidistant frontier clusters each step.
        prev_target = self._last_guidepost_target
        if prev_target is not None:
            prev_key = (prev_target[0], prev_target[1])
            incumbent_dist = dist_dict.get(prev_key, 1e8)
            if incumbent_dist >= 1e8 - 1:  # previous target no longer reachable/valid
                prev_target = None
                incumbent_dist = 1e8
        else:
            incumbent_dist = 1e8

        nearest_utility_coords = prev_target if prev_target is not None else robot_location
        switch_threshold = incumbent_dist * self._GUIDEPOST_HYSTERESIS
        for coords in utility_node_coords:
            dist = dist_dict[(coords[0], coords[1])]
            if 0 < dist < switch_threshold:
                switch_threshold = dist
                nearest_utility_coords = coords

        self._last_guidepost_target = nearest_utility_coords

        path_coords, dist = self.a_star(robot_location, nearest_utility_coords)
        guidepost = np.zeros_like(utility)
        for coords in path_coords:
            index = coord_to_index.get((coords[0], coords[1]))
            if index is not None:
                guidepost[index] = 1

        robot_in_graph = self.nodes_dict.nearest_neighbors(robot_location.tolist(), 1)[0].data.coords
        current_index = coord_to_index[(robot_in_graph[0], robot_in_graph[1])]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)

        # Modified: Only mark current robot's position in occupancy (no global position sharing)
        occupancy = np.zeros((n_nodes, 1))
        occupancy[current_index] = -1  # Mark only current robot's position

        return all_node_coords, utility, guidepost, occupancy, adjacent_matrix, current_index, neighbor_indices, highest_utility_angle, frontiers_distribution, heading_visited, visited_by_others, path_coords, visited_self

    def get_total_utility(self):
        total_utility = 0.0
        for node in self.nodes_dict.__iter__():
            total_utility += float(node.data.utility)
        return total_utility

    def Dijkstra(self, start):
        cache_key = (start[0], start[1])
        if cache_key in self._dijkstra_cache:
            return self._dijkstra_cache[cache_key]

        dist_dict = {}
        prev_dict = {}

        for node in self.nodes_dict.__iter__():
            coords = node.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None

        start_key = (start[0], start[1])
        assert start_key in dist_dict
        dist_dict[start_key] = 0

        heap = [(0.0, start_key)]
        visited = set()

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)

            node_entry = self.nodes_dict.find(u)
            if node_entry is None:
                print(u)
                for node in self.nodes_dict.__iter__():
                    print(node.data.coords)
                continue

            node = node_entry.data
            for neighbor_node_coords in node.neighbor_list:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in visited:
                    continue
                cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (
                        neighbor_node_coords[1] - u[1]) ** 2) ** (1 / 2)
                cost = np.round(cost, 2)
                alt = d + cost
                if alt < dist_dict.get(v, 1e8):
                    dist_dict[v] = alt
                    prev_dict[v] = u
                    heapq.heappush(heap, (alt, v))

        self._dijkstra_cache[cache_key] = (dist_dict, prev_dict)
        return dist_dict, prev_dict

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        if (end[0], end[1]) not in dist_dict:
            return [], 1e8

        dist = dist_dict[(end[0], end[1])]

        path = [(end[0], end[1])]
        prev_node = prev_dict[(end[0], end[1])]
        while prev_node is not None:
            path.append(prev_node)
            temp = prev_node
            prev_node = prev_dict[temp]

        path.reverse()
        return path[1:], np.round(dist, 2)

    def hop_distances_from(self, source):
        """BFS from source returning {node_key: hop_count} for all reachable nodes."""
        source_key = (source[0], source[1])
        if source_key in self._bfs_cache:
            return self._bfs_cache[source_key]
        if self.nodes_dict.find(source_key) is None:
            return {}
        hop_dict = {source_key: 0}
        queue = deque([source_key])
        while queue:
            u = queue.popleft()
            node_entry = self.nodes_dict.find(u)
            if node_entry is None:
                continue
            for neighbor_coords in node_entry.data.neighbor_list:
                v = (neighbor_coords[0], neighbor_coords[1])
                if v not in hop_dict:
                    hop_dict[v] = hop_dict[u] + 1
                    queue.append(v)
        self._bfs_cache[source_key] = hop_dict
        return hop_dict

    def h(self, coords_1, coords_2):
        h = ((coords_1[0] - coords_2[0]) ** 2 + (coords_1[1] - coords_2[1]) ** 2) ** (1 / 2)
        h = np.round(h, 2)
        return h

    def a_star(self, start, destination, boundary=None, max_dist=None):
        if not self.check_node_exist_in_dict(start):
            print(start)
            Warning("start position is not in node dict")
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            Warning("end position is not in node dict")
            return [], 1e8

        start_key = (start[0], start[1])
        dest_key = (destination[0], destination[1])

        if start_key == dest_key:
            return [destination], 0

        if boundary is None and max_dist is None:
            cache_key = (start_key, dest_key)
            if cache_key in self._astar_cache:
                return self._astar_cache[cache_key]

        g = {start_key: 0}
        parents = {start_key: start_key}
        closed_set = set()

        heap = [(self.h(start, destination), start_key)]

        while heap:
            _, n = heapq.heappop(heap)
            if n in closed_set:
                continue
            closed_set.add(n)

            if max_dist is not None and g[n] > max_dist:
                return [], 1e8

            node = self.nodes_dict.find(n).data
            n_coords = node.coords

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                curr = n
                while parents[curr] != curr:
                    path.append(curr)
                    curr = parents[curr]
                path.reverse()
                result = path, np.round(length, 2)
                if boundary is None and max_dist is None:
                    self._astar_cache[(start_key, dest_key)] = result
                return result

            for neighbor_node_coords in node.neighbor_list:
                if self.nodes_dict.find(neighbor_node_coords.tolist()) is None:
                    continue
                if boundary is not None:
                    if not (boundary[0] < neighbor_node_coords[0] < boundary[2] and boundary[1] < neighbor_node_coords[1] < boundary[3]):
                        continue
                cost = ((neighbor_node_coords[0] - n_coords[0]) ** 2 + (
                            neighbor_node_coords[1] - n_coords[1]) ** 2) ** (1 / 2)
                cost = np.round(cost, 2)
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                tentative_g = g[n] + cost
                if m not in g or tentative_g < g[m]:
                    g[m] = tentative_g
                    parents[m] = n
                    if m not in closed_set:
                        heapq.heappush(heap, (tentative_g + self.h(neighbor_node_coords, destination), m))

        print('Path does not exist!')

        result = [], 1e8
        if boundary is None and max_dist is None:
            self._astar_cache[(start_key, dest_key)] = result
        return result

class Node:
    def __init__(self, coords, frontiers, updating_map_info, fov, sensor_range, utility_range=None):
        self.coords = coords
        self.utility_range = utility_range
        self.utility = 0
        self.fov = fov
        self.highest_utility_angle = -360
        self.num_angles_bin = NUM_ANGLES_BIN
        self.sensor_range = sensor_range
        self.frontiers_distribution = np.zeros(self.num_angles_bin)
        self.heading_visited = np.zeros(self.num_angles_bin)
        self.observable_frontiers = self.initialize_observable_frontiers(frontiers, updating_map_info)
        self.visited = 0
        self.visited_by_others = 0.0  # Float: decays over time

        self.neighbor_matrix = -np.ones((NUM_NODE_NEIGHBORS, NUM_NODE_NEIGHBORS))
        self.neighbor_list = []
        self.neighbor_list.append(self.coords)
        self.need_update_neighbor = True

    def initialize_observable_frontiers(self, frontiers, updating_map_info):
        if len(frontiers) == 0:
            self.utility = 0
            return set()
        else:
            observable_frontiers = set()
            frontiers = np.array(list(frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
            new_frontiers_in_range = frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, updating_map_info)
                if not collision:
                    observable_frontiers.add((point[0], point[1]))
            self.utility = len(observable_frontiers)
            if self.utility <= MIN_UTILITY:
                self.utility = 0
                self.highest_utility_angle = -360
                observable_frontiers = set()
            else:
                # Calculate angles for all observable frontiers
                observable_frontiers_array = np.array(list(observable_frontiers))
                angles = np.degrees(np.arctan2(observable_frontiers_array[:, 1] - self.coords[1], 
                                               observable_frontiers_array[:, 0] - self.coords[0]) % (2 * np.pi))

                for angle in angles:
                    index = int(angle / 360 * self.num_angles_bin) % self.num_angles_bin
                    self.frontiers_distribution[index] += 1
              
                if angles.shape[0] == 1:
                    optimal_angle = angles[0]
                else:
                    half_fov_size = int((self.fov / 360)*self.num_angles_bin/2)
                    window = np.concatenate((self.frontiers_distribution[-half_fov_size:], self.frontiers_distribution, self.frontiers_distribution[:half_fov_size]))
                    indices = np.arange(len(self.frontiers_distribution)) + half_fov_size
                    sum_vector = np.sum(np.take(window, indices.reshape(-1, 1) + np.arange(-half_fov_size, half_fov_size + 1)), axis=1)
                    optimal_angle = np.argsort(-sum_vector)[0] * (360/self.num_angles_bin)
               
                if optimal_angle > 360:
                    optimal_angle = optimal_angle - 360
                    optimal_angle = 0 if optimal_angle == 360 else optimal_angle

                self.highest_utility_angle = optimal_angle

            return observable_frontiers

    def update_neighbor_nodes(self, updating_map_info, nodes_dict):
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    center_index = self.neighbor_matrix.shape[0] // 2
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                          self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, updating_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_list.append(neighbor_coords)

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_list.append(self.coords)

        if -1 not in self.neighbor_matrix:
            self.need_update_neighbor = False

    def update_node_observable_frontiers(self, frontiers, updating_map_info, map_info):
        self.frontiers_distribution = np.zeros(self.num_angles_bin)
        frontiers_observed = []
        for frontier in self.observable_frontiers:
            if not is_frontier(np.array([frontier[0], frontier[1]]), map_info):
                frontiers_observed.append(frontier)
        for frontier in frontiers_observed:
            self.observable_frontiers.remove(frontier)
        new_frontiers = frontiers - self.observable_frontiers
        new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)
        dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
        new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
        for point in new_frontiers_in_range:
            collision = check_collision(self.coords, point, updating_map_info)
            if not collision:
                self.observable_frontiers.add((point[0], point[1]))

        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()
            self.highest_utility_angle = -360
        else:
            observable_frontiers_array = np.array(list(self.observable_frontiers))
            angles = np.degrees(np.arctan2(observable_frontiers_array[:, 1] - self.coords[1], 
                                            observable_frontiers_array[:, 0] - self.coords[0]) % (2 * np.pi))
            for angle in angles:
                index = int(angle / 360 * self.num_angles_bin) % self.num_angles_bin
                self.frontiers_distribution[index] += 1
            
            if angles.shape[0] == 1:
                optimal_angle = angles[0]
            else:
                half_fov_size = int((self.fov / 360)*self.num_angles_bin/2)
                window = np.concatenate((self.frontiers_distribution[-half_fov_size:], self.frontiers_distribution, self.frontiers_distribution[:half_fov_size]))
                indices = np.arange(len(self.frontiers_distribution)) + half_fov_size
                sum_vector = np.sum(np.take(window, indices.reshape(-1, 1) + np.arange(-half_fov_size, half_fov_size + 1)), axis=1)
                optimal_angle = np.argsort(-sum_vector)[0] * (360/self.num_angles_bin)
            if optimal_angle > 360:
                optimal_angle = optimal_angle - 360
                optimal_angle = 0 if optimal_angle == 360 else optimal_angle

            self.highest_utility_angle = optimal_angle

    def set_visited(self, heading):
        self.visited = 1
        index = int(heading / 360 * self.num_angles_bin) % self.num_angles_bin
        fov_size = int((self.fov / 360)*self.num_angles_bin)
        start_index = int(index - fov_size/2)
        end_index = int(index + fov_size/2)
        if start_index < 0:
            self.heading_visited[0:end_index] = 1
            self.heading_visited[abs(start_index):] = 1
        elif end_index >= self.num_angles_bin:
            self.heading_visited[start_index:] = 1
            self.heading_visited[:abs(end_index)] = 1
        else:
            self.heading_visited[start_index:end_index] = 1
        self.need_update_neighbor = False
