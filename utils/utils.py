import numpy as np
import imageio
import os
import subprocess
import tempfile
from skimage.morphology import label

from parameter import *

DEFAULT_VIDEO_FPS = 10


def get_cell_position_from_coords(coords, map_info, check_negative=True):
    single_cell = False
    if coords.flatten().shape[0] == 2:
        single_cell = True

    coords = coords.reshape(-1, 2)
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    cell_x = ((coords_x - map_info.map_origin_x) / map_info.cell_size)
    cell_y = ((coords_y - map_info.map_origin_y) / map_info.cell_size)

    cell_position = np.around(np.stack((cell_x, cell_y), axis=-1)).astype(int)

    if check_negative:
        assert sum(cell_position.flatten() >= 0) == cell_position.flatten().shape[0], print(cell_position, coords, map_info.map_origin_x, map_info.map_origin_y)
    if single_cell:
        return cell_position[0]
    else:
        return cell_position

def get_coords_from_cell_position(cell_position, map_info):
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    cell_y = cell_position[:, 1]
    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == OCCUPIED:
        return coords[0]
    else:
        return coords

def get_free_area_coords(map_info):
    free_indices = np.where(map_info.map == FREE)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    free_coords = get_coords_from_cell_position(free_cells, map_info)
    return free_coords


def get_free_and_connected_map(location, map_info):
    free = (map_info.map == FREE).astype(float)
    labeled_free = label(free, connectivity=2)
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    connected_free_map = (labeled_free == label_number)
    return connected_free_map

def get_updating_node_coords(location, updating_map_info, check_connectivity=True):
    x_min = updating_map_info.map_origin_x
    y_min = updating_map_info.map_origin_y
    x_max = updating_map_info.map_origin_x + (updating_map_info.map.shape[1] - 1) * CELL_SIZE
    y_max = updating_map_info.map_origin_y + (updating_map_info.map.shape[0] - 1) * CELL_SIZE

    if x_min % NODE_RESOLUTION != 0:
        x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if x_max % NODE_RESOLUTION != 0:
        x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
    if y_min % NODE_RESOLUTION != 0:
        y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if y_max % NODE_RESOLUTION != 0:
        y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

    x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
    y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
    t1, t2 = np.meshgrid(x_coords, y_coords)
    nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    nodes = np.around(nodes, 1)

    free_connected_map = None

    if not check_connectivity:

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < updating_map_info.map.shape[0] and 0 <= cell[0] < updating_map_info.map.shape[1]
            if updating_map_info.map[cell[1], cell[0]] == FREE:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    else:
        free_connected_map = get_free_and_connected_map(location, updating_map_info)
        free_connected_map = np.array(free_connected_map)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < free_connected_map.shape[0] and 0 <= cell[0] < free_connected_map.shape[1]
            if free_connected_map[cell[1], cell[0]] == 1:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    return nodes, free_connected_map

def get_frontier_in_map(map_info):
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    unknown = (map_info.map == UNKNOWN) * 1
    unknown = np.lib.pad(unknown, ((1, 1), (1, 1)), 'constant', constant_values=0)
    unknown_neighbor = unknown[2:][:, 1:x_len + 1] + unknown[:y_len][:, 1:x_len + 1] + unknown[1:y_len + 1][:, 2:] \
                       + unknown[1:y_len + 1][:, :x_len] + unknown[:y_len][:, 2:] + unknown[2:][:, :x_len] + \
                       unknown[2:][:, 2:] + unknown[:y_len][:, :x_len]
    free_cell_indices = np.where(map_info.map.ravel(order='F') == FREE)[0]
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order='F'))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order='F') < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices]

    frontier_coords = get_coords_from_cell_position(frontier_cell, map_info).reshape(-1, 2)
    if frontier_cell.shape[0] > 0 and FRONTIER_CELL_SIZE != CELL_SIZE:
        frontier_coords = frontier_coords.reshape(-1 ,2)
        frontier_coords = frontier_down_sample(frontier_coords)
    else:
        frontier_coords = set(map(tuple, frontier_coords))
    return frontier_coords

def frontier_down_sample(data, voxel_size=FRONTIER_CELL_SIZE):
    voxel_indices = np.array(data / voxel_size, dtype=int).reshape(-1, 2)

    voxel_dict = {}
    for i, point in enumerate(data):
        voxel_index = tuple(voxel_indices[i])

        if voxel_index not in voxel_dict:
            voxel_dict[voxel_index] = point
        else:
            current_point = voxel_dict[voxel_index]
            if np.linalg.norm(point - np.array(voxel_index) * voxel_size) < np.linalg.norm(
                    current_point - np.array(voxel_index) * voxel_size):
                voxel_dict[voxel_index] = point

    downsampled_data = set(map(tuple, voxel_dict.values()))
    return downsampled_data

def is_frontier(location, map_info):
    cell = get_cell_position_from_coords(location, map_info)
    if map_info.map[cell[1], cell[0]] != FREE:
        return False
    else:
        assert cell[1] - 1 > 0 and cell[1] - 1 > 0 and cell[1] + 2 < map_info.map.shape[1] and cell[0] + 2 < map_info.map.shape[0]
        unknwon = map_info.map[cell[1] - 1:cell[1] + 2, cell[0] - 1: cell[0] + 2] == UNKNOWN
        n = np.sum(unknwon)
        if 1 < n < 8:
            return True
        else:
            return False

def _get_line_traversal_cells(start_cell, end_cell):
    x0, y0 = int(start_cell[0]), int(start_cell[1])
    x1, y1 = int(end_cell[0]), int(end_cell[1])

    cells = [(x0, y0)]
    dx = x1 - x0
    dy = y1 - y0
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    abs_dx = abs(dx)
    abs_dy = abs(dy)

    if abs_dx == 0 and abs_dy == 0:
        return cells

    t_delta_x = np.inf if abs_dx == 0 else 1.0 / abs_dx
    t_delta_y = np.inf if abs_dy == 0 else 1.0 / abs_dy
    t_max_x = np.inf if abs_dx == 0 else 0.5 * t_delta_x
    t_max_y = np.inf if abs_dy == 0 else 0.5 * t_delta_y

    x, y = x0, y0
    while x != x1 or y != y1:
        if np.isclose(t_max_x, t_max_y):
            next_x = x + step_x
            next_y = y + step_y
            cells.append((next_x, y))
            cells.append((x, next_y))
            x, y = next_x, next_y
            cells.append((x, y))
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            x += step_x
            cells.append((x, y))
            t_max_x += t_delta_x
        else:
            y += step_y
            cells.append((x, y))
            t_max_y += t_delta_y

    return cells

def check_collision(start, end, map_info, allow_unknown=False):
    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    traversed_cells = _get_line_traversal_cells(start_cell, end_cell)
    occupancy_map = map_info.map

    for x, y in traversed_cells:
        if not (0 <= x < occupancy_map.shape[1] and 0 <= y < occupancy_map.shape[0]):
            return True
        if x == end_cell[0] and y == end_cell[1]:
            continue

        cell_value = int(occupancy_map[y, x])
        if cell_value == OCCUPIED:
            return True
        if cell_value == UNKNOWN and not allow_unknown:
            return True

    return False


def _prepare_video_frame(image):
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    height, width = image.shape[:2]
    pad_height = height % 2
    pad_width = width % 2
    if pad_height or pad_width:
        image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='edge')

    return np.ascontiguousarray(image)


def _write_mp4(output_path, frame_files, fps=DEFAULT_VIDEO_FPS):
    output_dir = os.path.dirname(output_path) or '.'
    with tempfile.TemporaryDirectory(prefix='mp4_frames_', dir=output_dir) as tmp_dir:
        for idx, frame in enumerate(frame_files):
            image = imageio.imread(frame)
            prepared_frame = _prepare_video_frame(image)
            imageio.imwrite(os.path.join(tmp_dir, f'frame_{idx:06d}.png'), prepared_frame)

        command = [
            'ffmpeg',
            '-y',
            '-framerate', str(fps),
            '-i', os.path.join(tmp_dir, 'frame_%06d.png'),
            '-c:v', 'mpeg4',
            '-pix_fmt', 'yuv420p',
            output_path,
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def make_video(path, n, frame_files, rate):
    output_path = '{}/{}_explored_rate_{:.4g}.mp4'.format(path, n, rate)
    _write_mp4(output_path, frame_files, fps=DEFAULT_VIDEO_FPS)
    print('mp4 complete\n')

    for filename in frame_files[:-1]:
        os.remove(filename)


def make_video_test(path, n, frame_files, rate, n_agents, fov, sensor_range):
    output_path = '{}/{}_{}_{}_{}_explored_rate_{:.4g}.mp4'.format(path, n, n_agents, fov, sensor_range, rate)
    _write_mp4(output_path, frame_files, fps=DEFAULT_VIDEO_FPS)
    print('mp4 complete\n')

    for filename in frame_files[:-1]:
        os.remove(filename)


class MapInfo:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size

    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
