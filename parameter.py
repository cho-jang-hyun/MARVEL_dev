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

FOLDER_NAME = 'budget_urgency_new_2'
LOAD_FOLDER_NAME = 'budget_aware_reward_with_budget_urgency'
model_path = f'model/{FOLDER_NAME}' # save checkpoint
load_path = f'load_model/{LOAD_FOLDER_NAME}' # load checkpoint
train_path = f'train/{FOLDER_NAME}' # save tensorboard
gifs_path = f'gifs/{FOLDER_NAME}' # save gif

# save training data
SUMMARY_WINDOW = 32
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 1000
NUM_EPISODE_BUFFER = 46

# Sim parameters
N_AGENTS = 4
USE_CONTINUOUS_SIM = True
NUM_SIM_STEPS = 6
VELOCITY = 1
YAW_RATE = 35 # in degrees
SUCCESS_THRESHOLD = 0.95  # Episode ends when explored_rate >= this value

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
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
LR = 1e-4
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.001  # Soft update coefficient for target network (0.001 ~ 0.01)
NUM_META_AGENT = 12

# Gradient clipping parameters
GRAD_CLIP_POLICY = 1.0  # Max gradient norm for policy network (typical: 0.5 ~ 5.0)
GRAD_CLIP_Q = 10.0      # Max gradient norm for Q networks (typical: 1.0 ~ 10.0)

# network parameters
NODE_INPUT_DIM = 12  # Changed from 10: added initial_budget/BUDGET as mission scale context feature, plus [exploration_urgency, return_urgency] goal-vector

EMBEDDING_DIM = 128

# Trajectory tracking parameters
TRAJECTORY_HISTORY_LENGTH = 10  # Number of recent steps to track
TRAJECTORY_FEATURE_DIM = 5      # (dx, dy, sin(heading), cos(heading), velocity)
TRAJECTORY_EMBEDDING_DIM = 64   # Trajectory encoder output dimension
MAX_DETECTED_AGENTS = N_AGENTS - 1  # Maximum number of detectable agents in FOV
GATED_ATTENTION = True  # Use gated attention for cross attention (True: gated, False: standard residual)

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
TRAIN_ALGO = 3
# 0: SAC, 1:MAAC , 2: Ground Truth 3: MAAC and Ground Truth

# Communication settings
USE_COMMUNICATION = False  # True: MAAC with all agent communication (centralized critic)
                           # False: Decentralized learning with only FOV-based trajectory observation
                           # When False, agents only use their own observation + detected trajectories in FOV
                           # This simulates no-communication scenario where agents rely on visual detection only

# Budget and RTB parameters
BUDGET = MAX_EPISODE_STEP       # Max possible budget (used as normalization reference)
BUDGET_MIN_P1 = int(BUDGET * 0.75)  # Phase 1: light randomization (75-100%)
BUDGET_MIN_P2 = int(BUDGET * 0.50)  # Phase 2: full randomization  (50-100%)


RTB_REWARD_SCALE = 5.0
RTB_SUCCESS_BONUS = 20.0
RTB_BUDGET_BONUS = 5.0
RTB_EXHAUSTION_PENALTY = -10.0
RTB_TEAM_SURVIVAL_BONUS = 15.0
CURRICULUM_STEP1 = 2000
CURRICULUM_STEP2 = 4000
