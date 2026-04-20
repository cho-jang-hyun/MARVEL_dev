"""
Configuration parameters for marvel testing.

This file defines various parameters for:
- Model loading and testing configuration
- Simulation parameters
- Heading and navigation settings
- Map representation and resolution
- Sensor and utility thresholds
- Network and graph parameters
- GPU usage configuration

"""
TEST_SET = 'maps_test'
LOAD_FOLDER_NAME = 'MARVEL2_4_12'
CHECKPOINT_NAME = 'latest.pth'
load_path = f'load_model/{LOAD_FOLDER_NAME}/{CHECKPOINT_NAME}'
gifs_path = f'results/gifs/{LOAD_FOLDER_NAME}'
LOAD_MODEL = True  
SAVE_IMG_GAP = 50
SAVE_GIFS = False
GREEDY = True
NUM_RUN = 1
NUM_TEST = 100

# Sim parameters
USE_CONTINUOUS_SIM = True
NUM_SIM_STEPS = 6
MAX_EPISODE_STEP = 128
BUDGET_TIMESTEP_METERS = 8.0
BUDGET_TIMESTEPS = MAX_EPISODE_STEP
TEST_BUDGET_TIMESTEPS_LIST = [70, 80, 90, 100, 128]
BUDGET = BUDGET_TIMESTEPS * BUDGET_TIMESTEP_METERS
MAX_BUDGET = BUDGET
RETURN_SAFETY_MARGIN = 0.0
NUM_EPISODE_BUFFER = 54
VELOCITY = 1
YAW_RATE = 35 # in degrees
SUCCESS_THRESHOLD = 0.99

# Heading parameters
NUM_ANGLES_BIN = 36
NUM_HEADING_CANDIDATES = 3

# Map and planning resolution
CELL_SIZE = 0.4  # meter
NODE_RESOLUTION = 4.0  # meter
FRONTIER_CELL_SIZE = 2 * CELL_SIZE

# Map representation
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# Sensor and utility range
MIN_UTILITY = 1
VISITED_BY_OTHERS_DECAY = 0.025
VISITED_BY_OTHERS_MIN = 0.05

# Updating map range w.r.t the robot
UPDATING_MAP_SIZE = 15 * NODE_RESOLUTION

# Testing parameters
NUM_META_AGENT = 16
INITIAL_EXPLORED_RATE = 0.90

# Network parameters
NODE_INPUT_DIM = 9  # +1 for visited_self (binary flag: agent has visited this node)
EMBEDDING_DIM = 128
BUDGET_FEATURE_DIM = 4  # log_initial_m, log_remaining_m, remaining/initial, distance_to_base/remaining
USE_TRAJECTORY = True  # Enable trajectory encoder
GATED_ATTENTION = True

# Trajectory tracking parameters (same as parameter.py)
TRAJECTORY_HISTORY_LENGTH = 10  # Number of recent steps to track
TRAJECTORY_FEATURE_DIM = 5      # (dx, dy, sin(heading), cos(heading), age_since_seen)
TRAJECTORY_EMBEDDING_DIM = 64   # Trajectory encoder output dimension
# N_AGENTS will be determined dynamically in test, so we use max value
MAX_DETECTED_AGENTS = 10  # Maximum number of detectable agents in FOV (conservative estimate)

# Graph parameters
NUM_NODE_NEIGHBORS = 5
K_SIZE = NUM_NODE_NEIGHBORS**2   # the number of neighboring nodes
NODE_PADDING_SIZE = 360

# GPU usage
USE_GPU = True # False
NUM_GPU = 0

# Communication settings (same as parameter.py)
USE_COMMUNICATION = False  # True: Use all agent communication
                           # False: Decentralized testing with only FOV-based trajectory observation
