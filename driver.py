"""
Main driver script for training MARVEL.

This script initializes and manages a distributed training process using Ray, 
implementing a Soft Actor-Critic (SAC) algorithm with multiple meta agents. 
Key functionalities include:
- Initializing neural networks for policy and Q-value estimation
- Setting up distributed training with multiple workers
- Managing experience replay buffer
- Performing training iterations 
- Logging metrics and saving model checkpoints

The main training loop handles:
- Collecting experiences from distributed workers
- Sampling and training on batches of experiences
- Updating policy, Q-networks, and temperature parameter
- Periodic model checkpointing and performance logging

Supports GPU/CPU training, WandB logging, and model resumption.
"""
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
import wandb

from utils.model import PolicyNet, QNet
from utils.runner import RLRunner
from parameter import *

ray.init()
print("Welcome to MARVEL!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

if not os.path.exists(load_path):
    os.makedirs(load_path)


def load_compatible_state_dict(module, checkpoint_state, label):
    model_state = module.state_dict()
    compatible_state = {
        key: value for key, value in checkpoint_state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    missing_or_mismatched = len(model_state) - len(compatible_state)
    module.load_state_dict(compatible_state, strict=False)
    print(f'Loaded {label}: {len(compatible_state)} tensors, skipped {missing_or_mismatched}')


def load_optimizer_state_if_compatible(optimizer, checkpoint_state, label):
    try:
        optimizer.load_state_dict(checkpoint_state)
    except ValueError as exc:
        print(f'Skipped {label} optimizer state due to architecture change: {exc}')


def safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return np.nan
    return float(np.nanmean(values))


class RewardNormalizer:
    """Running reward normalization using Welford's online algorithm.

    Normalizes rewards to zero-mean, unit-variance using an exponential
    moving estimate of mean and variance.  This keeps Q-value magnitudes
    bounded and prevents gradient explosion in the critic.
    """

    def __init__(self, momentum: float = 0.001, eps: float = 1e-6):
        self.momentum = momentum
        self.eps = eps
        self.running_mean = 0.0
        self.running_var = 1.0

    def normalize(self, reward_tensor):
        """Normalize a batch of rewards in-place on the same device."""
        with torch.no_grad():
            batch_mean = reward_tensor.mean().item()
            batch_var = reward_tensor.var().item()
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            std = max(self.running_var ** 0.5, self.eps)
            return (reward_tensor - self.running_mean) / std

    def state_dict(self):
        return {'running_mean': self.running_mean, 'running_var': self.running_var}

    def load_state_dict(self, state):
        self.running_mean = state.get('running_mean', 0.0)
        self.running_var = state.get('running_var', 1.0)


def snapshot_module_state(module, target_device):
    """Snapshot model weights for distribution to Ray workers.

    Uses .data to avoid autograd overhead and skips the redundant .clone()
    when .to() already returns a new tensor (i.e. when crossing devices).
    """
    state = module.state_dict()
    if target_device == next(module.parameters()).device:
        # Same device: .to() is a no-op, so we must clone explicitly.
        return {k: v.data.clone() for k, v in state.items()}
    # Cross-device: .to() already allocates a new tensor; clone is wasteful.
    return {k: v.data.to(target_device) for k, v in state.items()}


def collect_rollouts(experience_buffer, sample_indices):
    rollouts = []
    for i in range(len(experience_buffer)):
        if len(experience_buffer[i]) == 0:
            rollouts.append([])
        else:
            rollouts.append([experience_buffer[i][index] for index in sample_indices])
    return rollouts


def sample_phase_stratified_indices(indices, phase_entries, batch_size):
    if len(phase_entries) == 0:
        return random.sample(indices, batch_size)

    post_indices = []
    pre_indices = []
    for idx in indices:
        phase_value = float(phase_entries[idx].item())
        if phase_value > 0.5:
            post_indices.append(idx)
        else:
            pre_indices.append(idx)

    if len(post_indices) < POST_PHASE_REPLAY_MIN_SAMPLES or len(pre_indices) == 0:
        return random.sample(indices, batch_size)

    target_post = int(round(batch_size * POST_PHASE_BATCH_RATIO))
    target_post = max(1, min(batch_size - 1, target_post))
    target_pre = batch_size - target_post

    if len(post_indices) >= target_post:
        sampled_post = random.sample(post_indices, target_post)
    else:
        sampled_post = random.choices(post_indices, k=target_post)

    if len(pre_indices) >= target_pre:
        sampled_pre = random.sample(pre_indices, target_pre)
    else:
        sampled_pre = random.choices(pre_indices, k=target_pre)

    sampled_indices = sampled_pre + sampled_post
    random.shuffle(sampled_indices)
    return sampled_indices


def _stack_to(rollout_list, device, non_blocking=True):
    """Stack a list of tensors on CPU, then transfer once to target device."""
    stacked = torch.stack(rollout_list)
    if stacked.device == device:
        return stacked
    return stacked.to(device, non_blocking=non_blocking)


def prepare_training_batch(rollouts, device, effective_train_algo):
    batch = {}

    # Core actor observation (indices 0-7) — stack on CPU, single transfer
    _s = _stack_to
    batch['node_inputs'] = _s(rollouts[0], device)
    batch['node_padding_mask'] = _s(rollouts[1], device)
    batch['local_edge_mask'] = _s(rollouts[2], device)
    batch['current_local_index'] = _s(rollouts[3], device)
    batch['current_local_edge'] = _s(rollouts[4], device)
    batch['local_edge_padding_mask'] = _s(rollouts[5], device)
    batch['frontier_distribution'] = _s(rollouts[6], device)
    batch['heading_visited'] = _s(rollouts[7], device)
    batch['action'] = _s(rollouts[8], device)
    batch['reward'] = _s(rollouts[9], device)
    batch['done'] = _s(rollouts[10], device)
    batch['next_node_inputs'] = _s(rollouts[11], device)
    batch['next_node_padding_mask'] = _s(rollouts[12], device)
    batch['next_local_edge_mask'] = _s(rollouts[13], device)
    batch['next_current_local_index'] = _s(rollouts[14], device)
    batch['next_current_local_edge'] = _s(rollouts[15], device)
    batch['next_local_edge_padding_mask'] = _s(rollouts[16], device)
    batch['next_frontier_distribution'] = _s(rollouts[17], device)
    batch['next_heading_visited'] = _s(rollouts[18], device)
    batch['neighbor_best_headings'] = _s(rollouts[38], device)
    batch['next_neighbor_best_headings'] = _s(rollouts[39], device)
    batch['budget_state'] = _s(rollouts[48], device)
    batch['next_budget_state'] = _s(rollouts[50], device)
    batch['detected_trajectories'] = _s(rollouts[40], device)
    batch['trajectory_mask'] = _s(rollouts[41], device)
    batch['trajectory_node_indices'] = _s(rollouts[42], device)
    batch['next_detected_trajectories'] = _s(rollouts[43], device)
    batch['next_trajectory_mask'] = _s(rollouts[44], device)
    batch['next_trajectory_node_indices'] = _s(rollouts[45], device)

    if len(rollouts[52]) > 0:
        batch['critic_phase_flag'] = _s(rollouts[52], device)
        batch['next_critic_phase_flag'] = _s(rollouts[53], device)
    else:
        zero_phase = torch.zeros_like(batch['done'], dtype=torch.float32, device=device)
        batch['critic_phase_flag'] = zero_phase
        batch['next_critic_phase_flag'] = zero_phase.clone()

    batch['observation'] = [
        batch['node_inputs'], batch['node_padding_mask'], batch['local_edge_mask'], batch['current_local_index'],
        batch['current_local_edge'], batch['local_edge_padding_mask'], batch['frontier_distribution'], batch['heading_visited'],
        batch['neighbor_best_headings'], batch['budget_state']
    ]
    batch['next_observation'] = [
        batch['next_node_inputs'], batch['next_node_padding_mask'], batch['next_local_edge_mask'], batch['next_current_local_index'],
        batch['next_current_local_edge'], batch['next_local_edge_padding_mask'], batch['next_frontier_distribution'],
        batch['next_heading_visited'], batch['next_neighbor_best_headings'], batch['next_budget_state']
    ]

    batch['policy_kwargs'] = dict(
        detected_trajectories=batch['detected_trajectories'],
        trajectory_mask=batch['trajectory_mask'],
        trajectory_node_indices=batch['trajectory_node_indices'],
    )
    batch['next_policy_kwargs'] = dict(
        detected_trajectories=batch['next_detected_trajectories'],
        trajectory_mask=batch['next_trajectory_mask'],
        trajectory_node_indices=batch['next_trajectory_node_indices'],
    )

    if effective_train_algo in (2, 3, 4, 5):
        batch['critic_node_inputs'] = _s(rollouts[19], device)
        batch['critic_node_padding_mask'] = _s(rollouts[20], device)
        batch['critic_edge_mask'] = _s(rollouts[21], device)
        batch['critic_current_index'] = _s(rollouts[22], device)
        batch['critic_current_edge'] = _s(rollouts[23], device)
        batch['critic_edge_padding_mask'] = _s(rollouts[24], device)
        batch['critic_frontier_distribution'] = _s(rollouts[25], device)
        batch['critic_heading_visited'] = _s(rollouts[26], device)
        batch['critic_next_node_inputs'] = _s(rollouts[27], device)
        batch['critic_next_node_padding_mask'] = _s(rollouts[28], device)
        batch['critic_next_edge_mask'] = _s(rollouts[29], device)
        batch['critic_next_current_index'] = _s(rollouts[30], device)
        batch['critic_next_current_edge'] = _s(rollouts[31], device)
        batch['critic_next_edge_padding_mask'] = _s(rollouts[32], device)
        batch['critic_next_frontier_distribution'] = _s(rollouts[33], device)
        batch['critic_next_heading_visited'] = _s(rollouts[34], device)

    if effective_train_algo in (4, 5):
        batch['critic_neighbor_best_headings'] = _s(rollouts[46], device)
        batch['next_critic_neighbor_best_headings'] = _s(rollouts[47], device)
        batch['critic_trajectory_node_indices'] = _s(rollouts[49], device)
        batch['next_critic_trajectory_node_indices'] = _s(rollouts[51], device)

    if effective_train_algo in (1, 3, 5):
        batch['all_agent_indices'] = _s(rollouts[35], device)
        batch['all_agent_next_indices'] = _s(rollouts[36], device)
        batch['next_all_agent_next_indices'] = _s(rollouts[37], device)

    if effective_train_algo == 0:
        batch['state'] = batch['observation']
        batch['next_state'] = batch['next_observation']
        batch['q_kwargs'] = dict(
            phase_flag=batch['critic_phase_flag'],
            detected_trajectories=batch['detected_trajectories'],
            trajectory_mask=batch['trajectory_mask'],
            trajectory_node_indices=batch['trajectory_node_indices'],
        )
        batch['next_q_kwargs'] = dict(
            phase_flag=batch['next_critic_phase_flag'],
            detected_trajectories=batch['next_detected_trajectories'],
            trajectory_mask=batch['next_trajectory_mask'],
            trajectory_node_indices=batch['next_trajectory_node_indices'],
        )
    elif effective_train_algo == 1:
        batch['state'] = batch['observation']
        batch['next_state'] = batch['next_observation']
        batch['q_kwargs'] = dict(
            all_agent_indices=batch['all_agent_indices'],
            all_agent_next_indices=batch['all_agent_next_indices'],
            phase_flag=batch['critic_phase_flag'],
            detected_trajectories=batch['detected_trajectories'],
            trajectory_mask=batch['trajectory_mask'],
            trajectory_node_indices=batch['trajectory_node_indices'],
        )
        batch['next_q_kwargs'] = dict(
            all_agent_indices=batch['all_agent_next_indices'],
            all_agent_next_indices=batch['next_all_agent_next_indices'],
            phase_flag=batch['next_critic_phase_flag'],
            detected_trajectories=batch['next_detected_trajectories'],
            trajectory_mask=batch['next_trajectory_mask'],
            trajectory_node_indices=batch['next_trajectory_node_indices'],
        )
    elif effective_train_algo == 2:
        batch['state'] = [
            batch['critic_node_inputs'], batch['critic_node_padding_mask'], batch['critic_edge_mask'], batch['critic_current_index'],
            batch['critic_current_edge'], batch['critic_edge_padding_mask'], batch['critic_frontier_distribution'], batch['critic_heading_visited'],
            batch['neighbor_best_headings'], batch['budget_state']
        ]
        batch['next_state'] = [
            batch['critic_next_node_inputs'], batch['critic_next_node_padding_mask'], batch['critic_next_edge_mask'],
            batch['critic_next_current_index'], batch['critic_next_current_edge'], batch['critic_next_edge_padding_mask'],
            batch['critic_next_frontier_distribution'], batch['critic_next_heading_visited'], batch['next_neighbor_best_headings'],
            batch['next_budget_state']
        ]
        batch['q_kwargs'] = dict(
            phase_flag=batch['critic_phase_flag'],
            detected_trajectories=batch['detected_trajectories'],
            trajectory_mask=batch['trajectory_mask'],
            trajectory_node_indices=None,
        )
        batch['next_q_kwargs'] = dict(
            phase_flag=batch['next_critic_phase_flag'],
            detected_trajectories=batch['next_detected_trajectories'],
            trajectory_mask=batch['next_trajectory_mask'],
            trajectory_node_indices=None,
        )
    elif effective_train_algo == 3:
        batch['state'] = [
            batch['critic_node_inputs'], batch['critic_node_padding_mask'], batch['critic_edge_mask'], batch['critic_current_index'],
            batch['critic_current_edge'], batch['critic_edge_padding_mask'], batch['critic_frontier_distribution'], batch['critic_heading_visited'],
            batch['neighbor_best_headings'], batch['budget_state']
        ]
        batch['next_state'] = [
            batch['critic_next_node_inputs'], batch['critic_next_node_padding_mask'], batch['critic_next_edge_mask'],
            batch['critic_next_current_index'], batch['critic_next_current_edge'], batch['critic_next_edge_padding_mask'],
            batch['critic_next_frontier_distribution'], batch['critic_next_heading_visited'], batch['next_neighbor_best_headings'],
            batch['next_budget_state']
        ]
        batch['q_kwargs'] = dict(
            all_agent_indices=batch['all_agent_indices'],
            all_agent_next_indices=batch['all_agent_next_indices'],
            phase_flag=batch['critic_phase_flag'],
            detected_trajectories=batch['detected_trajectories'],
            trajectory_mask=batch['trajectory_mask'],
            trajectory_node_indices=None,
        )
        batch['next_q_kwargs'] = dict(
            all_agent_indices=batch['all_agent_next_indices'],
            all_agent_next_indices=batch['next_all_agent_next_indices'],
            phase_flag=batch['next_critic_phase_flag'],
            detected_trajectories=batch['next_detected_trajectories'],
            trajectory_mask=batch['next_trajectory_mask'],
            trajectory_node_indices=None,
        )
    elif effective_train_algo == 4:
        batch['state'] = [
            batch['critic_node_inputs'], batch['critic_node_padding_mask'], batch['critic_edge_mask'], batch['critic_current_index'],
            batch['critic_current_edge'], batch['critic_edge_padding_mask'], batch['critic_frontier_distribution'], batch['critic_heading_visited'],
            batch['critic_neighbor_best_headings'], batch['budget_state']
        ]
        batch['next_state'] = [
            batch['critic_next_node_inputs'], batch['critic_next_node_padding_mask'], batch['critic_next_edge_mask'],
            batch['critic_next_current_index'], batch['critic_next_current_edge'], batch['critic_next_edge_padding_mask'],
            batch['critic_next_frontier_distribution'], batch['critic_next_heading_visited'], batch['next_critic_neighbor_best_headings'],
            batch['next_budget_state']
        ]
        batch['q_kwargs'] = dict(
            phase_flag=batch['critic_phase_flag'],
            detected_trajectories=batch['detected_trajectories'],
            trajectory_mask=batch['trajectory_mask'],
            trajectory_node_indices=batch['critic_trajectory_node_indices'],
        )
        batch['next_q_kwargs'] = dict(
            phase_flag=batch['next_critic_phase_flag'],
            detected_trajectories=batch['next_detected_trajectories'],
            trajectory_mask=batch['next_trajectory_mask'],
            trajectory_node_indices=batch['next_critic_trajectory_node_indices'],
        )
    elif effective_train_algo == 5:
        batch['state'] = [
            batch['critic_node_inputs'], batch['critic_node_padding_mask'], batch['critic_edge_mask'], batch['critic_current_index'],
            batch['critic_current_edge'], batch['critic_edge_padding_mask'], batch['critic_frontier_distribution'], batch['critic_heading_visited'],
            batch['critic_neighbor_best_headings'], batch['budget_state']
        ]
        batch['next_state'] = [
            batch['critic_next_node_inputs'], batch['critic_next_node_padding_mask'], batch['critic_next_edge_mask'],
            batch['critic_next_current_index'], batch['critic_next_current_edge'], batch['critic_next_edge_padding_mask'],
            batch['critic_next_frontier_distribution'], batch['critic_next_heading_visited'], batch['next_critic_neighbor_best_headings'],
            batch['next_budget_state']
        ]
        batch['q_kwargs'] = dict(
            all_agent_indices=batch['all_agent_indices'],
            all_agent_next_indices=batch['all_agent_next_indices'],
            phase_flag=batch['critic_phase_flag'],
            detected_trajectories=batch['detected_trajectories'],
            trajectory_mask=batch['trajectory_mask'],
            trajectory_node_indices=batch['critic_trajectory_node_indices'],
        )
        batch['next_q_kwargs'] = dict(
            all_agent_indices=batch['all_agent_next_indices'],
            all_agent_next_indices=batch['next_all_agent_next_indices'],
            phase_flag=batch['next_critic_phase_flag'],
            detected_trajectories=batch['next_detected_trajectories'],
            trajectory_mask=batch['next_trajectory_mask'],
            trajectory_node_indices=batch['next_critic_trajectory_node_indices'],
        )
    else:
        raise ValueError(f"Unsupported effective_train_algo: {effective_train_algo}")

    return batch


def main():
    # Set specific GPU if GPU_ID is specified
    if USE_GPU_GLOBAL and GPU_ID is not None:
        torch.cuda.set_device(GPU_ID)
        print(f"Using GPU {GPU_ID}: {torch.cuda.get_device_name(GPU_ID)}")

    # use GPU/CPU for driver/worker
    device = torch.device(f'cuda:{GPU_ID}') if USE_GPU_GLOBAL and GPU_ID is not None else torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    # Determine effective training algorithm based on communication setting
    # When USE_COMMUNICATION=False, disable agent communication in QNet
    if USE_COMMUNICATION:
        if TRAIN_ALGO == 4:
            effective_train_algo = 5
        else:
            effective_train_algo = TRAIN_ALGO
    else:
        # Remove agent communication component from TRAIN_ALGO
        # TRAIN_ALGO 3 (MAAC + GT) -> 2 (GT only)
        # TRAIN_ALGO 1 (MAAC) -> 0 (SAC)
        if TRAIN_ALGO == 3:
            effective_train_algo = 2  # Ground Truth only, no communication
        elif TRAIN_ALGO == 1:
            effective_train_algo = 0  # SAC, no communication
        elif TRAIN_ALGO == 4:
            effective_train_algo = 4  # Merged-belief critic, no communication
        else:
            effective_train_algo = TRAIN_ALGO  # 0 or 2 already have no communication

    # target entropy for SAC (discrete action space) updated
    action_dim = K_SIZE * NUM_HEADING_CANDIDATES
    entropy_target = 0.5 * np.log(action_dim)
    log_alpha_min = -4.0   # raised from -8.0; alpha=e^-4≈0.018 maintains minimum exploration (was collapsing to e^-7≈0.0007)
    log_alpha_max = 2.0

    print(f"Training Configuration:")
    print(f"  TRAIN_ALGO: {TRAIN_ALGO}")
    print(f"  USE_COMMUNICATION: {USE_COMMUNICATION}")
    print(f"  Effective TRAIN_ALGO for QNet: {effective_train_algo}")
    print(f"  Using Trajectory Encoder: True")
    print(f"  Using Gated Attention: {GATED_ATTENTION}")
    print(f"  Gradient Clipping - Policy: {GRAD_CLIP_POLICY}, Q: {GRAD_CLIP_Q}")
    print(f"  Entropy Target: {entropy_target:.4f} (action_dim={action_dim})")

    # initialize neural networks
    global_policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, use_trajectory=True, gated_attention=GATED_ATTENTION).to(device)
    global_q_net1 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, effective_train_algo, use_trajectory=True, gated_attention=GATED_ATTENTION).to(device)
    global_q_net2 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, effective_train_algo, use_trajectory=True, gated_attention=GATED_ATTENTION).to(device)
    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True

    global_target_q_net1 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, effective_train_algo, use_trajectory=True, gated_attention=GATED_ATTENTION).to(device)
    global_target_q_net2 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, effective_train_algo, use_trajectory=True, gated_attention=GATED_ATTENTION).to(device)

    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

    reward_normalizer = RewardNormalizer(momentum=0.001)

    curr_episode = 0
    best_success_rate = -float('inf')  # Track best performance for saving best model

    if USE_WANDB:
        import parameter
        vars(parameter).__delitem__('__builtins__')
        wandb.init(project='Directional', name=FOLDER_NAME, config=vars(parameter), resume='allow',
                   id=None, notes=None)
        wandb.watch([global_policy_net, global_q_net1], log='all', log_freq=1000, log_graph=False)


    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(load_path + '/latest.pth', map_location=device)
        load_compatible_state_dict(global_policy_net, checkpoint['policy_model'], 'policy')
        load_compatible_state_dict(global_q_net1, checkpoint['q_net1_model'], 'q_net1')
        load_compatible_state_dict(global_q_net2, checkpoint['q_net2_model'], 'q_net2')
        log_alpha = checkpoint['log_alpha'].to(device)
        log_alpha.requires_grad_(True)
        log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

        load_optimizer_state_if_compatible(global_policy_optimizer, checkpoint['policy_optimizer'], 'policy')
        load_optimizer_state_if_compatible(global_q_net1_optimizer, checkpoint['q_net1_optimizer'], 'q_net1')
        load_optimizer_state_if_compatible(global_q_net2_optimizer, checkpoint['q_net2_optimizer'], 'q_net2')
        load_optimizer_state_if_compatible(log_alpha_optimizer, checkpoint['log_alpha_optimizer'], 'log_alpha')
        curr_episode = checkpoint['episode']
        best_success_rate = checkpoint.get('best_success_rate', -float('inf'))
        if 'reward_normalizer' in checkpoint:
            reward_normalizer.load_state_dict(checkpoint['reward_normalizer'])

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get global networks weights
    weights_set = []
    policy_weights = snapshot_module_state(global_policy_net, local_device)
    weights_set.append(policy_weights)

    # distributed training if multiple GPUs available
    if NUM_GPU > 1 and USE_GPU_GLOBAL:
        device_ids = list(range(NUM_GPU))
        dp_policy = nn.DataParallel(global_policy_net, device_ids=device_ids)
        dp_q_net1 = nn.DataParallel(global_q_net1, device_ids=device_ids)
        dp_q_net2 = nn.DataParallel(global_q_net2, device_ids=device_ids)
        dp_target_q_net1 = nn.DataParallel(global_target_q_net1, device_ids=device_ids)
        dp_target_q_net2 = nn.DataParallel(global_target_q_net2, device_ids=device_ids)
    else:
        # Single GPU or CPU - no DataParallel needed
        dp_policy = global_policy_net
        dp_q_net1 = global_q_net1
        dp_q_net2 = global_q_net2
        dp_target_q_net1 = global_target_q_net1
        dp_target_q_net2 = global_target_q_net2

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))

    # initialize metric collector
    metric_name = ['merged_travel_dist', 'travel_dist', 'success_rate', 'explored_rate', 'episode_reward']
    training_data = []
    current_success_rate = -float('inf')
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(NUM_EPISODE_BUFFER):
        experience_buffer.append([])

    # collect data from worker and do training
    try:
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list)
            # get the results
            done_jobs = ray.get(done_id)

            # save experience and metric
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            # launch new task
            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))

            # start training
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("Training model!")

                # keep the replay buffer size
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                # training for n times each step
                for j in range(4):
                    actor_sample_indices = random.sample(indices, BATCH_SIZE)
                    actor_rollouts = collect_rollouts(experience_buffer, actor_sample_indices)
                    actor_batch = prepare_training_batch(actor_rollouts, device, effective_train_algo)

                    critic_sample_indices = sample_phase_stratified_indices(
                        indices,
                        experience_buffer[52] if len(experience_buffer) > 52 else [],
                        BATCH_SIZE,
                    )
                    critic_rollouts = collect_rollouts(experience_buffer, critic_sample_indices)
                    critic_batch = prepare_training_batch(critic_rollouts, device, effective_train_algo)

                    # SAC
                    with torch.no_grad():
                        q_values1 = dp_q_net1(*actor_batch['state'], **actor_batch['q_kwargs'])
                        q_values2 = dp_q_net2(*actor_batch['state'], **actor_batch['q_kwargs'])
                        q_values = torch.min(q_values1, q_values2)

                    logp = dp_policy(*actor_batch['observation'], **actor_batch['policy_kwargs'])
                    policy_loss = torch.sum(
                        (logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach())),
                        dim=1).mean()

                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=GRAD_CLIP_POLICY,
                                                                      norm_type=2)
                    global_policy_optimizer.step()

                    # Normalize rewards to stabilize Q-value magnitudes and prevent gradient explosion
                    raw_reward = critic_batch['reward']
                    normalized_reward = reward_normalizer.normalize(raw_reward)

                    with torch.no_grad():
                        next_logp = dp_policy(*critic_batch['next_observation'], **critic_batch['next_policy_kwargs'])
                        next_q_values1 = dp_target_q_net1(*critic_batch['next_state'], **critic_batch['next_q_kwargs'])
                        next_q_values2 = dp_target_q_net2(*critic_batch['next_state'], **critic_batch['next_q_kwargs'])
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime = torch.sum(
                            next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)),
                            dim=1).unsqueeze(1)
                        target_q_batch = normalized_reward + GAMMA * (1 - critic_batch['done']) * value_prime

                    mse_loss = nn.MSELoss()

                    q_values1 = dp_q_net1(*critic_batch['state'], **critic_batch['q_kwargs'])
                    q1 = torch.gather(q_values1, 1, critic_batch['action'])
                    q1_loss = mse_loss(q1, target_q_batch.detach()).mean()
                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=GRAD_CLIP_Q,
                                                                 norm_type=2)
                    global_q_net1_optimizer.step()

                    q_values2 = dp_q_net2(*critic_batch['state'], **critic_batch['q_kwargs'])
                    q2 = torch.gather(q_values2, 1, critic_batch['action'])
                    q2_loss = mse_loss(q2, target_q_batch.detach()).mean()
                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=GRAD_CLIP_Q,
                                                                 norm_type=2)
                    global_q_net2_optimizer.step()

                    entropy = (logp.exp() * logp).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy.detach() - entropy_target)).mean()

                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()
                    with torch.no_grad():
                        log_alpha.clamp_(log_alpha_min, log_alpha_max)

                    # Soft update target networks
                    for target_param, param in zip(global_target_q_net1.parameters(), global_q_net1.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                    for target_param, param in zip(global_target_q_net2.parameters(), global_q_net2.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(safe_nanmean(perf_metrics[n]))

                policy_model = dp_policy.module if hasattr(dp_policy, 'module') else dp_policy
                trajectory_debug = getattr(getattr(policy_model, 'trajectory_encoder', None), 'latest_debug', {})
                traj_detected_agents = float(trajectory_debug.get('detected_agents_mean', torch.tensor(0.0)).item())
                traj_usable_agents = float(trajectory_debug.get('usable_agents_mean', torch.tensor(0.0)).item())
                traj_valid_timestep_ratio = float(trajectory_debug.get('valid_timestep_ratio', torch.tensor(0.0)).item())
                traj_embedding_norm = float(trajectory_debug.get('embedding_norm', torch.tensor(0.0)).item())
                traj_attention_entropy = float(trajectory_debug.get('agent_attention_entropy', torch.tensor(0.0)).item())

                critic_phase_post_ratio = 0.0
                critic_next_phase_post_ratio = 0.0
                critic_pre_q_mean = 0.0
                critic_post_q_mean = 0.0
                critic_active_q_mean = 0.0
                critic_head_gap = 0.0
                if effective_train_algo in (2, 3, 4, 5):
                    critic_model = dp_q_net1.module if hasattr(dp_q_net1, 'module') else dp_q_net1
                    with torch.no_grad():
                        q_pre_debug, q_post_debug = critic_model(
                            *critic_batch['state'],
                            return_both_heads=True,
                            **critic_batch['q_kwargs'],
                        )
                        selected_q_pre = torch.gather(q_pre_debug, 1, critic_batch['action'])
                        selected_q_post = torch.gather(q_post_debug, 1, critic_batch['action'])
                        phase_mask = critic_batch['critic_phase_flag'].reshape(critic_batch['critic_phase_flag'].size(0), 1, 1).float()
                        selected_active_q = (1.0 - phase_mask) * selected_q_pre + phase_mask * selected_q_post

                    critic_phase_post_ratio = float(critic_batch['critic_phase_flag'].float().mean().item())
                    critic_next_phase_post_ratio = float(critic_batch['next_critic_phase_flag'].float().mean().item())
                    critic_pre_q_mean = float(selected_q_pre.mean().item())
                    critic_post_q_mean = float(selected_q_post.mean().item())
                    critic_active_q_mean = float(selected_active_q.mean().item())
                    critic_head_gap = float((selected_q_post - selected_q_pre).abs().mean().item())

                # Use the actual rollout episode reward (from perf_metrics) instead of the
                # replay-buffer batch mean, which is a lagging/biased indicator that masks
                # real performance degradation.
                rollout_reward = safe_nanmean(perf_metrics.get('episode_reward', []))

                data = [rollout_reward, value_prime.mean().item(), policy_loss.item(), q1_loss.item(),
                        entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(),
                        alpha_loss.item(), traj_detected_agents, traj_usable_agents, traj_valid_timestep_ratio,
                        traj_embedding_norm, traj_attention_entropy, critic_phase_post_ratio,
                        critic_next_phase_post_ratio, critic_pre_q_mean, critic_post_q_mean,
                        critic_active_q_mean, critic_head_gap, *perf_data]
                training_data.append(data)
                current_success_rate = np.nanmean(perf_metrics['success_rate']) if len(perf_metrics['success_rate']) > 0 else -float('inf')

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                write_to_tensor_board(writer, training_data, curr_episode)
                writer.add_scalar('BudgetSampling/min_budget_meters', MIN_BUDGET, curr_episode)
                writer.add_scalar('BudgetSampling/max_budget_meters', MAX_BUDGET, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            weights_set = []
            policy_weights = snapshot_module_state(global_policy_net, local_device)
            weights_set.append(policy_weights)

            # save the model
            if curr_episode % 32 == 0 or curr_episode % 1000 == 0:
                should_save_best = current_success_rate > best_success_rate
                if should_save_best:
                    best_success_rate = current_success_rate

                print('Saving model', end='\n')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
                              "q_net1_model": global_q_net1.state_dict(),
                              "q_net2_model": global_q_net2.state_dict(),
                              "log_alpha": log_alpha,
                              "policy_optimizer": global_policy_optimizer.state_dict(),
                              "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                              "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                              "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
                              "episode": curr_episode,
                              "best_success_rate": best_success_rate,
                              "reward_normalizer": reward_normalizer.state_dict(),
                              }

                # Save latest model (always)
                path_latest = "./" + model_path + "/latest.pth"
                torch.save(checkpoint, path_latest)
                print(f'Saved latest model (episode {curr_episode})')

                # Save checkpoint every 1000 steps
                if curr_episode % 1000 == 0:
                    path_checkpoint = "./" + model_path + f"/checkpoint_{curr_episode}.pth"
                    torch.save(checkpoint, path_checkpoint)
                    print(f'Saved checkpoint_{curr_episode}.pth')

                # Save best model if performance improved
                if should_save_best:
                    path_best = "./" + model_path + "/best.pth"
                    torch.save(checkpoint, path_best)
                    print(f'Saved best model (success_rate: {best_success_rate:.4f})')

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
        ray.shutdown()
        if USE_WANDB:
            wandb.finish(quiet=True)

def write_to_tensor_board(writer, tensorboard_data, curr_episode):
    tensorboard_data = np.atleast_2d(np.asarray(tensorboard_data, dtype=float))
    tensorboard_data = [
        safe_nanmean(tensorboard_data[:, column_idx])
        for column_idx in range(tensorboard_data.shape[1])
    ]
    (rollout_reward, value, policy_loss, q_value_loss, entropy, policy_grad_norm,
     q_value_grad_norm, log_alpha, alpha_loss, traj_detected_agents,
     traj_usable_agents, traj_valid_timestep_ratio, traj_embedding_norm,
     traj_attention_entropy, critic_phase_post_ratio, critic_next_phase_post_ratio,
     critic_pre_q_mean, critic_post_q_mean, critic_active_q_mean, critic_head_gap,
     merged_travel_dist, travel_dist, success_rate, explored_rate,
     episode_reward) = tensorboard_data

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policy_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Alpha Loss', scalar_value=alpha_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Loss', scalar_value=q_value_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Grad Norm', scalar_value=policy_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Grad Norm', scalar_value=q_value_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Log Alpha', scalar_value=log_alpha, global_step=curr_episode)
    writer.add_scalar(tag='Trajectory/Detected Agents', scalar_value=traj_detected_agents, global_step=curr_episode)
    writer.add_scalar(tag='Trajectory/Usable Agents', scalar_value=traj_usable_agents, global_step=curr_episode)
    writer.add_scalar(tag='Trajectory/Valid Timestep Ratio', scalar_value=traj_valid_timestep_ratio, global_step=curr_episode)
    writer.add_scalar(tag='Trajectory/Embedding Norm', scalar_value=traj_embedding_norm, global_step=curr_episode)
    writer.add_scalar(tag='Trajectory/Agent Attention Entropy', scalar_value=traj_attention_entropy, global_step=curr_episode)
    writer.add_scalar(tag='Critic/Phase Post Ratio', scalar_value=critic_phase_post_ratio, global_step=curr_episode)
    writer.add_scalar(tag='Critic/Next Phase Post Ratio', scalar_value=critic_next_phase_post_ratio, global_step=curr_episode)
    writer.add_scalar(tag='Critic/Selected Pre Head Q Mean', scalar_value=critic_pre_q_mean, global_step=curr_episode)
    writer.add_scalar(tag='Critic/Selected Post Head Q Mean', scalar_value=critic_post_q_mean, global_step=curr_episode)
    writer.add_scalar(tag='Critic/Selected Active Head Q Mean', scalar_value=critic_active_q_mean, global_step=curr_episode)
    writer.add_scalar(tag='Critic/Selected Head Gap', scalar_value=critic_head_gap, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=rollout_reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Episode Reward', scalar_value=episode_reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Merged Objective Travel Distance', scalar_value=merged_travel_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Full Episode Travel Distance', scalar_value=travel_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Travel Distance', scalar_value=travel_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Explored Rate', scalar_value=explored_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)

if __name__ == "__main__":
    main()
