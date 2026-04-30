"""
Configuration parameters for MARVEL simulation and training.

This module defines key parameters for:
- Folder and path configurations
- Simulation settings
- Drone and sensor characteristics 
- Map representation
- Training hyperparameters
- Neural network architecture
- Computational resource settings

Key configurations include:
- Number of agents
- Sensor ranges
- Map resolution
- Episode and training parameters
- GPU and logging options
"""

FOLDER_NAME = '4_30'
LOAD_FOLDER_NAME = '4_22_Dual_critic_estimation_metric_with_indv_complete_reward'
model_path = f'model/{FOLDER_NAME}' # save checkpoint
load_path = f'load_model/{LOAD_FOLDER_NAME}' # load checkpoint
train_path = f'train/{FOLDER_NAME}' # save tensorboard
gifs_path = f'gifs/{FOLDER_NAME}' # save gif

# save training data
SUMMARY_WINDOW = 32
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 1000
NUM_EPISODE_BUFFER = 54

# Sim parameters
N_AGENTS = 4
USE_CONTINUOUS_SIM = True
NUM_SIM_STEPS = 6
VELOCITY = 1
YAW_RATE = 35 # in degrees
SUCCESS_THRESHOLD = 0.99  # Coverage-based completion threshold for mission and local return logic

# Heading parameters
FOV = 120   # in degrees
V_FOV = 60
MOUNTING_ANGLE = 15 # downwards
NUM_ANGLES_BIN = 36
NUM_HEADING_CANDIDATES = 3
DRONE_HEIGHT = 2

# map and planning resolution
CELL_SIZE = 0.4  # meter
NODE_RESOLUTION = 4.0  # meter
FRONTIER_CELL_SIZE = 2 * CELL_SIZE

# map representation
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# sensor and utility range
SENSOR_RANGE = 10  # meter
UTILITY_RANGE = 0.9 * SENSOR_RANGE
MIN_UTILITY = 1

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION

# training parameters
MAX_EPISODE_STEP = 128
BUDGET_TIMESTEP_METERS = 8.0
BUDGET_START_TIMESTEPS = 128
BUDGET_END_TIMESTEPS = 80
BUDGET_CURRICULUM_NOISE_TIMESTEPS = 16
BUDGET_START = BUDGET_START_TIMESTEPS * BUDGET_TIMESTEP_METERS   # budget when success rate is 0 (easy), in meters
BUDGET_END = BUDGET_END_TIMESTEPS * BUDGET_TIMESTEP_METERS       # budget when success rate is 1 (hard), in meters
BUDGET_CURRICULUM_NOISE = BUDGET_CURRICULUM_NOISE_TIMESTEPS * BUDGET_TIMESTEP_METERS   # ±uniform noise around the curriculum target budget, in meters
BUDGET_CURRICULUM_EMA = 0.002  # EMA smoothing for success rate tracker
BUDGET_CURRICULUM_UNIFORM_P = 0.6  # probability of ignoring curriculum and sampling uniformly (prevents forgetting)
BUDGET_CURRICULUM_SUCCESS_THRESHOLD = 0.995  # stricter success signal used only for curriculum updates
BUDGET_CURRICULUM_WARMUP_EPISODES = 1500  # keep near-high budgets before curriculum starts decaying
BUDGET_CURRICULUM_UPDATE_STRIDE = 8  # apply EMA after aggregating this many completed episodes
BUDGET = BUDGET_START
MIN_BUDGET = BUDGET_END
MAX_BUDGET = BUDGET_START
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 8000
BATCH_SIZE = 128
LR = 2e-5
GAMMA = 0.99
TAU = 0.003  # Soft update coefficient for target network (0.001 ~ 0.01)
NUM_META_AGENT = 15
POST_PHASE_REPLAY_MIN_SAMPLES = 512
POST_PHASE_BATCH_RATIO = 0.3

# reward shaping
MERGED_NODE_UTILITY_REWARD_WEIGHT = 0.0
VISITED_BY_OTHERS_DECAY = 0.025
VISITED_BY_OTHERS_MIN = 0.08
LOW_UTILITY_MOVE_THRESHOLD = 0.05
REPEATED_LOW_UTILITY_PENALTY = 0.05
TIME_PRESSURE_WEIGHT = 0.05  # per-step cost that scales linearly as budget depletes (0 at full budget, this value at zero budget)
MERGED_SUCCESS_BONUS = 5.0  # one-time bonus when merged explored coverage reaches SUCCESS_THRESHOLD (reduced from 6.5 to limit TD target spikes and critic gradient instability)
INDIVIDUAL_SUCCESS_BONUS = 2.5  # one-time bonus when an agent's individual explored coverage exceeds SUCCESS_THRESHOLD

# Gradient clipping parameters
GRAD_CLIP_POLICY = 1.0  # Max gradient norm for policy network (typical: 0.5 ~ 5.0)
GRAD_CLIP_Q = 20.0  # Max gradient norm for Q networks (raised from 1.5; was being clipped 100x causing near-random critic updates)

# network parameters
NODE_INPUT_DIM = 9  # Local actor node features, including visited_self as the final scalar
EMBEDDING_DIM = 128
BUDGET_FEATURE_DIM = 4  # log_initial_m, log_remaining_m, remaining/initial, distance_to_base/remaining
RETURN_SAFETY_MARGIN = 0.0  # extra meters reserved before an agent must stop exploring and head home
SIMULATE_RETURN_TO_BASE = False  # Returning agents are simulated while teammates still explore; once all agents are returning, the episode stops.

# Trajectory tracking parameters
TRAJECTORY_HISTORY_LENGTH = 10  # Number of recent steps to track
TRAJECTORY_FEATURE_DIM = 5      # (dx, dy, sin(heading), cos(heading), age_since_seen)
TRAJECTORY_EMBEDDING_DIM = 64   # Trajectory encoder output dimension
MAX_DETECTED_AGENTS = N_AGENTS - 1  # Maximum number of detectable agents in FOV
GATED_ATTENTION = True  # Apply paper-style query-dependent sigmoid gating on each attention head output

# Graph parameters
NUM_NODE_NEIGHBORS = 5
K_SIZE = NUM_NODE_NEIGHBORS**2   # the number of neighboring nodes
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value

# GPU usage
USE_GPU = False  # Workers use CPU to avoid GPU memory conflict with Ray
USE_GPU_GLOBAL = True  # Main training process uses GPU
NUM_GPU = 1  # Number of GPUs for DataParallel in main process
GPU_ID = 0  # Which GPU to use (0 or 1). Set to None to use all available GPUs

USE_WANDB = False
TRAIN_ALGO = 0
# 0: SAC
# 1: MAAC
# 2: Ground Truth critic
# 3: MAAC + Ground Truth critic
# 4: Merged-belief critic with teammate-only delta

# Communication settings
USE_COMMUNICATION = False  # True: MAAC with all agent communication (centralized critic)
                           # False: Decentralized learning with only FOV-based trajectory observation
                           # When False, agents only use their own observation + detected trajectories in FOV
                           # This simulates no-communication scenario where agents rely on visual detection only
