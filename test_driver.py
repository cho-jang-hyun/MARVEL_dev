"""
Runs distributed testing for a multi-agent exploration policy using Ray.

This function initializes a distributed testing framework where multiple meta-agents 
run test episodes with varying parameters such as number of agents, field of view, 
and sensor range. It collects and aggregates performance metrics across different 
test configurations.

Key operations:
- Loads a pre-trained policy network
- Distributes test jobs across multiple Ray workers
- Runs tests with different experimental parameters
- Collects and prints performance metrics including:
  - Travel distance
  - Exploration rate
  - Success rate
  - Overlap ratio

The function supports GPU acceleration and allows configurable testing parameters.
"""
import ray
import numpy as np
import torch
import os
import time
import warnings
from tqdm import tqdm
import json

os.environ.setdefault('MARVEL_CONFIG_MODE', 'test')

from utils.model import PolicyNet
from utils.test_worker import TestWorker
from utils.runtime_config import *
import csv

EVAL_USE_GPU = USE_GPU and NUM_GPU > 0 and torch.cuda.is_available()
BUDGET_EXPLORATION_RECORD_PATH = "record_10agents_budget_exploration.csv"
BOXPLOT_METRICS_RECORD_PATH = "record_10agents_boxplot_metrics.csv"

if USE_GPU and NUM_GPU > 0 and not torch.cuda.is_available():
    warnings.warn("USE_GPU is True but CUDA is unavailable; falling back to CPU.")


def load_compatible_state_dict(module, checkpoint_state, label):
    model_state = module.state_dict()
    compatible_state = {
        key: value for key, value in checkpoint_state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    missing_or_mismatched = len(model_state) - len(compatible_state)
    module.load_state_dict(compatible_state, strict=False)
    print(f'Loaded {label}: {len(compatible_state)} tensors, skipped {missing_or_mismatched}')


def safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return np.nan
    return float(np.nanmean(values))


def safe_nanstd(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return np.nan
    return float(np.nanstd(values))


def _history_value(history, step, default=np.nan):
    if step < len(history):
        return history[step]
    return default


def _agent_history_value(history, step, agent_id, default=np.nan):
    if step >= len(history):
        return default
    step_values = history[step]
    if agent_id >= len(step_values):
        return default
    return step_values[agent_id]


def append_budget_exploration_rows(metrics, info, n_agent, fov, sensor_range, budget_timesteps):
    merged_history = metrics.get('merged_explored_rate_history', metrics.get('explored_rate_history', []))
    individual_history = metrics.get('individual_explored_rate_history', [])
    remaining_history = metrics.get('remaining_budget_history', [])
    initial_budgets = metrics.get('initial_budget_history', [])
    remaining_mean_history = metrics.get('remaining_budget_mean_history', [])
    remaining_min_history = metrics.get('remaining_budget_min_history', [])
    remaining_max_history = metrics.get('remaining_budget_max_history', [])
    length_history = metrics.get('length_history', [])
    overlap_history = metrics.get('overlap_ratio_history', [])
    breakdown_enabled = bool(metrics.get('breakdown_enabled', False))
    breakdown_agent_count = int(metrics.get('breakdown_agent_count', 0))
    breakdown_random_seed = int(metrics.get('breakdown_random_seed', 0))
    broken_agent_ids = json.dumps(metrics.get('broken_agent_ids', []), sort_keys=True)
    breakdown_scheduled_steps = json.dumps(metrics.get('breakdown_scheduled_steps', {}), sort_keys=True)
    breakdown_triggered_steps = json.dumps(metrics.get('breakdown_triggered_steps', {}), sort_keys=True)
    num_broken_agents_final = int(metrics.get('num_broken_agents_final', 0))

    num_steps = max(
        len(merged_history),
        len(individual_history),
        len(remaining_history),
        len(length_history),
        len(overlap_history),
    )
    if num_steps == 0:
        return 0

    budget_m = float(budget_timesteps) * float(BUDGET_TIMESTEP_METERS)
    mean_initial_budget = (
        float(np.mean(np.asarray(initial_budgets, dtype=float)))
        if len(initial_budgets) > 0 else budget_m
    )
    fieldnames = [
        'test_set',
        'episode_number',
        'step',
        'scope',
        'agent_id',
        'n_agent',
        'fov_deg',
        'sensor_range_m',
        'budget_timesteps',
        'budget_m',
        'remaining_budget_m',
        'consumed_budget_m',
        'explored_rate',
        'merged_explored_rate',
        'team_remaining_budget_mean_m',
        'team_remaining_budget_min_m',
        'team_remaining_budget_max_m',
        'max_travel_dist_m',
        'overlap_ratio',
        'breakdown_enabled',
        'breakdown_agent_count',
        'breakdown_random_seed',
        'broken_agent_ids',
        'breakdown_scheduled_steps',
        'breakdown_triggered_steps',
        'num_broken_agents_final',
    ]
    write_header = (
        not os.path.exists(BUDGET_EXPLORATION_RECORD_PATH)
        or os.path.getsize(BUDGET_EXPLORATION_RECORD_PATH) == 0
    )

    rows_written = 0
    with open(BUDGET_EXPLORATION_RECORD_PATH, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for step in range(num_steps):
            merged_rate = float(_history_value(merged_history, step))
            team_remaining_mean = float(_history_value(remaining_mean_history, step))
            team_remaining_min = float(_history_value(remaining_min_history, step))
            team_remaining_max = float(_history_value(remaining_max_history, step))
            max_travel_dist = float(_history_value(length_history, step))
            overlap_ratio = float(_history_value(overlap_history, step))

            common = {
                'test_set': TEST_SET,
                'episode_number': info.get('episode_number', np.nan),
                'step': step,
                'n_agent': n_agent,
                'fov_deg': fov,
                'sensor_range_m': sensor_range,
                'budget_timesteps': budget_timesteps,
                'budget_m': budget_m,
                'merged_explored_rate': merged_rate,
                'team_remaining_budget_mean_m': team_remaining_mean,
                'team_remaining_budget_min_m': team_remaining_min,
                'team_remaining_budget_max_m': team_remaining_max,
                'max_travel_dist_m': max_travel_dist,
                'overlap_ratio': overlap_ratio,
                'breakdown_enabled': breakdown_enabled,
                'breakdown_agent_count': breakdown_agent_count,
                'breakdown_random_seed': breakdown_random_seed,
                'broken_agent_ids': broken_agent_ids,
                'breakdown_scheduled_steps': breakdown_scheduled_steps,
                'breakdown_triggered_steps': breakdown_triggered_steps,
                'num_broken_agents_final': num_broken_agents_final,
            }

            writer.writerow({
                **common,
                'scope': 'merged',
                'agent_id': '',
                'remaining_budget_m': team_remaining_mean,
                'consumed_budget_m': mean_initial_budget - team_remaining_mean,
                'explored_rate': merged_rate,
            })
            rows_written += 1

            for agent_id in range(n_agent):
                agent_remaining = float(_agent_history_value(remaining_history, step, agent_id))
                agent_initial = (
                    float(initial_budgets[agent_id])
                    if agent_id < len(initial_budgets) else budget_m
                )
                agent_explored = float(_agent_history_value(individual_history, step, agent_id))
                writer.writerow({
                    **common,
                    'scope': 'individual',
                    'agent_id': agent_id,
                    'remaining_budget_m': agent_remaining,
                    'consumed_budget_m': agent_initial - agent_remaining,
                    'explored_rate': agent_explored,
                })
                rows_written += 1

    return rows_written


def _write_metric_row(writer, common, metric_name, metric_value, scope='episode', agent_id='', step=''):
    try:
        value = float(metric_value)
    except (TypeError, ValueError):
        return 0
    writer.writerow({
        **common,
        'step': step,
        'scope': scope,
        'agent_id': agent_id,
        'metric_name': metric_name,
        'metric_value': value,
    })
    return 1


def _write_metric_series(writer, common, metric_name, values, scope='step'):
    rows_written = 0
    for step, value in enumerate(values):
        rows_written += _write_metric_row(
            writer,
            common,
            metric_name,
            value,
            scope=scope,
            step=step,
        )
    return rows_written


def _write_agent_metric_series(writer, common, metric_name, values, scope='agent_step'):
    rows_written = 0
    for step, step_values in enumerate(values):
        for agent_id, value in enumerate(step_values):
            rows_written += _write_metric_row(
                writer,
                common,
                metric_name,
                value,
                scope=scope,
                agent_id=agent_id,
                step=step,
            )
    return rows_written


def append_boxplot_metric_rows(metrics, info, n_agent, fov, sensor_range, utility_range, budget_timesteps):
    fieldnames = [
        'test_set',
        'episode_number',
        'n_agent',
        'fov_deg',
        'sensor_range_m',
        'utility_range_m',
        'budget_timesteps',
        'budget_m',
        'step',
        'scope',
        'agent_id',
        'metric_name',
        'metric_value',
        'breakdown_enabled',
        'breakdown_agent_count',
        'breakdown_random_seed',
        'broken_agent_ids',
        'breakdown_scheduled_steps',
        'breakdown_triggered_steps',
        'num_broken_agents_final',
    ]
    write_header = (
        not os.path.exists(BOXPLOT_METRICS_RECORD_PATH)
        or os.path.getsize(BOXPLOT_METRICS_RECORD_PATH) == 0
    )
    budget_m = float(budget_timesteps) * float(BUDGET_TIMESTEP_METERS)
    common = {
        'test_set': TEST_SET,
        'episode_number': info.get('episode_number', np.nan),
        'n_agent': n_agent,
        'fov_deg': fov,
        'sensor_range_m': sensor_range,
        'utility_range_m': utility_range,
        'budget_timesteps': budget_timesteps,
        'budget_m': budget_m,
        'breakdown_enabled': bool(metrics.get('breakdown_enabled', False)),
        'breakdown_agent_count': int(metrics.get('breakdown_agent_count', 0)),
        'breakdown_random_seed': int(metrics.get('breakdown_random_seed', 0)),
        'broken_agent_ids': json.dumps(metrics.get('broken_agent_ids', []), sort_keys=True),
        'breakdown_scheduled_steps': json.dumps(metrics.get('breakdown_scheduled_steps', {}), sort_keys=True),
        'breakdown_triggered_steps': json.dumps(metrics.get('breakdown_triggered_steps', {}), sort_keys=True),
        'num_broken_agents_final': int(metrics.get('num_broken_agents_final', 0)),
    }

    rows_written = 0
    with open(BOXPLOT_METRICS_RECORD_PATH, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        episode_metrics = {
            'travel_dist_m': metrics.get('travel_dist', np.nan),
            'merged_travel_dist_m': metrics.get('merged_travel_dist', np.nan),
            'final_merged_explored_rate': metrics.get('explored_rate', np.nan),
            'success_rate_0_99': metrics.get('success_rate', np.nan),
            'dist_to_0_90_merged_m': metrics.get('dist_to_0_90', np.nan),
            'dist_to_0_99_merged_m': metrics.get('dist_to_0_99', np.nan),
            'individual_explored_rate_mean': metrics.get('individual_explored_rate_mean', np.nan),
            'individual_explored_rate_std': metrics.get('individual_explored_rate_std', np.nan),
            'compute_time_mean_s': metrics.get('compute_time_mean', np.nan),
            'compute_time_std_s': metrics.get('compute_time_std', np.nan),
        }
        for metric_name, metric_value in episode_metrics.items():
            rows_written += _write_metric_row(writer, common, metric_name, metric_value)

        agent_metrics = {
            'final_individual_explored_rate': metrics.get('individual_explored_rates', []),
            'dist_to_0_90_individual_m': metrics.get('individual_dist_to_0_90', []),
            'dist_to_0_99_individual_m': metrics.get('individual_dist_to_0_99', []),
            'initial_budget_m': metrics.get('initial_budget_history', []),
        }
        final_remaining_budget = metrics.get('remaining_budget_history', [])
        if len(final_remaining_budget) > 0:
            agent_metrics['final_remaining_budget_m'] = final_remaining_budget[-1]

        for metric_name, values in agent_metrics.items():
            for agent_id, value in enumerate(values):
                rows_written += _write_metric_row(
                    writer,
                    common,
                    metric_name,
                    value,
                    scope='agent',
                    agent_id=agent_id,
                )

        step_series = {
            'max_travel_dist_m': metrics.get('length_history', []),
            'merged_explored_rate': metrics.get('merged_explored_rate_history', metrics.get('explored_rate_history', [])),
            'overlap_ratio': metrics.get('overlap_ratio_history', []),
            'compute_time_s': metrics.get('compute_time_history', []),
            'remaining_budget_mean_m': metrics.get('remaining_budget_mean_history', []),
            'remaining_budget_min_m': metrics.get('remaining_budget_min_history', []),
            'remaining_budget_max_m': metrics.get('remaining_budget_max_history', []),
        }
        for metric_name, values in step_series.items():
            rows_written += _write_metric_series(writer, common, metric_name, values)

        rows_written += _write_agent_metric_series(
            writer,
            common,
            'individual_explored_rate',
            metrics.get('individual_explored_rate_history', []),
        )
        rows_written += _write_agent_metric_series(
            writer,
            common,
            'remaining_budget_m',
            metrics.get('remaining_budget_history', []),
        )

    return rows_written

def run_test():
    device = torch.device('cuda') if EVAL_USE_GPU else torch.device('cpu')
    global_network = PolicyNet(
        NODE_INPUT_DIM,
        EMBEDDING_DIM,
        NUM_ANGLES_BIN,
        use_trajectory=True,
        gated_attention=GATED_ATTENTION,
    ).to(device)

    if device.type == 'cuda':
        checkpoint = torch.load(f'{load_path}')
    else:
        checkpoint = torch.load(f'{load_path}', map_location=torch.device('cpu'))

    load_compatible_state_dict(global_network, checkpoint['policy_model'], 'policy')
    
    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()

    all_fov = [120]
    all_n_agent = [4]
    all_sensor_range = [10]
    all_utility_range = [range_val * 0.9 for range_val in all_sensor_range]
    all_budget_timesteps = TEST_BUDGET_TIMESTEPS_LIST

    for n_agent in all_n_agent:
        for fov in all_fov:
            for sensor_range, utility_range in zip(all_sensor_range, all_utility_range):
                for budget_timesteps in all_budget_timesteps:

                    curr_test = 0

                    dist_history = []
                    merged_dist_history = []
                    explore_rate = []
                    success_rate = []
                    dist_to_merged_0_90 = []
                    dist_to_merged_0_99 = []
                    individual_explored_rates = []
                    dist_to_individual_0_90 = []
                    dist_to_individual_0_99 = []
                    compute_time_values = []
                    all_length_history = []
                    all_explored_rate_history = []
                    all_overlap_ratio_history =[]
                    budget_exploration_rows = 0
                    boxplot_metric_rows = 0

                    job_list = []
                    progress_bar = tqdm(
                        total=NUM_TEST,
                        desc=f"n={n_agent}, fov={fov}, range={sensor_range}, budget={budget_timesteps}ts",
                        leave=True,
                    )
                    for i, meta_agent in enumerate(meta_agents):
                        job_list.append(meta_agent.job.remote(weights, curr_test, n_agent, fov, sensor_range, utility_range, budget_timesteps))
                        curr_test += 1

                    try:
                        while len(dist_history) < curr_test:
                            done_id, job_list = ray.wait(job_list)
                            done_jobs = ray.get(done_id)

                            for job in done_jobs:
                                metrics, info = job
                                dist_history.append(metrics['travel_dist'])
                                merged_dist_history.append(metrics.get('merged_travel_dist', np.nan))
                                explore_rate.append(metrics['explored_rate'])
                                success_rate.append(metrics['success_rate'])
                                dist_to_merged_0_90.append(metrics.get('dist_to_0_90', np.nan))
                                dist_to_merged_0_99.append(metrics.get('dist_to_0_99', np.nan))
                                individual_explored_rates.extend(metrics.get('individual_explored_rates', []))
                                dist_to_individual_0_90.extend(metrics.get('individual_dist_to_0_90', []))
                                dist_to_individual_0_99.extend(metrics.get('individual_dist_to_0_99', []))
                                compute_time_values.extend(metrics.get('compute_time_history', []))
                                all_length_history.extend(metrics['length_history'])
                                all_explored_rate_history.extend(metrics['explored_rate_history'])
                                all_overlap_ratio_history.extend(metrics['overlap_ratio_history'])
                                budget_exploration_rows += append_budget_exploration_rows(
                                    metrics,
                                    info,
                                    n_agent,
                                    fov,
                                    sensor_range,
                                    budget_timesteps,
                                )
                                boxplot_metric_rows += append_boxplot_metric_rows(
                                    metrics,
                                    info,
                                    n_agent,
                                    fov,
                                    sensor_range,
                                    utility_range,
                                    budget_timesteps,
                                )
                                progress_bar.update(1)

                                if curr_test < NUM_TEST:
                                    job_list.append(
                                        meta_agents[info['id']].job.remote(
                                            weights, curr_test, n_agent, fov, sensor_range, utility_range, budget_timesteps
                                        )
                                    )
                                    curr_test += 1

                        print('|#Test set:', TEST_SET)
                        print('|#Total test:', NUM_TEST)
                        print('|#Number of agents:', n_agent)
                        print('|#FOV (degrees):', fov)
                        print('|#Sensor range (m):', sensor_range)
                        print('|#Budget (timesteps):', budget_timesteps)
                        print('|#Budget (m):', budget_timesteps * BUDGET_TIMESTEP_METERS)
                        print('|#Ave length:', np.array(dist_history).mean())
                        print('|#Max length:', np.array(dist_history).max())
                        print('|#Min length:', np.array(dist_history).min())
                        print('|#Std length:', np.array(dist_history).std())
                        print('|#Ave explored rate:', np.array(explore_rate).mean())
                        print('|#Ave success rate (0.99):', np.array(success_rate).mean())
                        print('|#Ave dist to 0.90 merged explored rate:', safe_nanmean(dist_to_merged_0_90))
                        print('|#Std dist to 0.90 merged explored rate:', safe_nanstd(dist_to_merged_0_90))
                        print('|#Ave dist to 0.99 merged explored rate:', safe_nanmean(dist_to_merged_0_99))
                        print('|#Std dist to 0.99 merged explored rate:', safe_nanstd(dist_to_merged_0_99))
                        print('|#Ave individual explored rate:', safe_nanmean(individual_explored_rates))
                        print('|#Std individual explored rate:', safe_nanstd(individual_explored_rates))
                        print('|#Ave dist to 0.90 individual explored rate:', safe_nanmean(dist_to_individual_0_90))
                        print('|#Std dist to 0.90 individual explored rate:', safe_nanstd(dist_to_individual_0_90))
                        print('|#Ave dist to 0.99 individual explored rate:', safe_nanmean(dist_to_individual_0_99))
                        print('|#Std dist to 0.99 individual explored rate:', safe_nanstd(dist_to_individual_0_99))
                        print('|#Ave overlap ratio:', np.array(all_overlap_ratio_history).mean())
                        print('|#Std overlap ratio:', np.array(all_overlap_ratio_history).std())
                        print('|#Ave compute time:', safe_nanmean(compute_time_values))
                        print('|#Std compute time:', safe_nanstd(compute_time_values))
                        print('|#Budget/exploration CSV rows:', budget_exploration_rows)
                        print('|#Boxplot metrics CSV rows:', boxplot_metric_rows)
                        
                        lines = [
                        f"|#Test set: {TEST_SET}",
                        f"|#Total test: {NUM_TEST}",
                        f"|#Number of agents: {n_agent}",
                        f"|#FOV (degrees): {fov}",
                        f"|#Sensor range (m): {sensor_range}",
                        f"|#Budget (timesteps): {budget_timesteps}",
                        f"|#Budget (m): {budget_timesteps * BUDGET_TIMESTEP_METERS}",
                        f"|#Ave length: {np.array(dist_history).mean()}",
                        f"|#Max length: {np.array(dist_history).max()}",
                        f"|#Min length: {np.array(dist_history).min()}",
                        f"|#Std length: {np.array(dist_history).std()}",
                        f"|#Ave explored rate: {np.array(explore_rate).mean()}",
                        f"|#Ave success rate (0.99): {np.array(success_rate).mean()}",
                        f"|#Ave dist to 0.90 merged explored rate: {safe_nanmean(dist_to_merged_0_90)}",
                        f"|#Std dist to 0.90 merged explored rate: {safe_nanstd(dist_to_merged_0_90)}",
                        f"|#Ave dist to 0.99 merged explored rate: {safe_nanmean(dist_to_merged_0_99)}",
                        f"|#Std dist to 0.99 merged explored rate: {safe_nanstd(dist_to_merged_0_99)}",
                        f"|#Ave individual explored rate: {safe_nanmean(individual_explored_rates)}",
                        f"|#Std individual explored rate: {safe_nanstd(individual_explored_rates)}",
                        f"|#Ave dist to 0.90 individual explored rate: {safe_nanmean(dist_to_individual_0_90)}",
                        f"|#Std dist to 0.90 individual explored rate: {safe_nanstd(dist_to_individual_0_90)}",
                        f"|#Ave dist to 0.99 individual explored rate: {safe_nanmean(dist_to_individual_0_99)}",
                        f"|#Std dist to 0.99 individual explored rate: {safe_nanstd(dist_to_individual_0_99)}",
                        f"|#Ave overlap ratio: {np.array(all_overlap_ratio_history).mean()}",
                        f"|#Std overlap ratio: {np.array(all_overlap_ratio_history).std()}",
                        f"|#Ave compute time: {safe_nanmean(compute_time_values)}",
                        f"|#Std compute time: {safe_nanstd(compute_time_values)}",
                        f"|#Budget/exploration CSV: {BUDGET_EXPLORATION_RECORD_PATH}",
                        f"|#Budget/exploration CSV rows: {budget_exploration_rows}",
                        f"|#Boxplot metrics CSV: {BOXPLOT_METRICS_RECORD_PATH}",
                        f"|#Boxplot metrics CSV rows: {boxplot_metric_rows}",
                        ]

                        with open("record_10agents.txt", "a") as f:
                            for line in lines:
                                f.write(line + "\n")
                            f.write("\n")  # 실행 1회 구분용 빈 줄
                        progress_bar.close()

                    except KeyboardInterrupt:
                        print("CTRL_C pressed. Killing remote workers")
                        progress_bar.close()
                        for a in meta_agents:
                            ray.kill(a)


@ray.remote(num_cpus=1, num_gpus=(NUM_GPU/NUM_META_AGENT) if EVAL_USE_GPU else 0)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if EVAL_USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(
            NODE_INPUT_DIM,
            EMBEDDING_DIM,
            NUM_ANGLES_BIN,
            use_trajectory=True,
            gated_attention=GATED_ATTENTION,
        )
        self.local_network.to(self.device)

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number, n_agent, fov, sensor_range, utility_range, budget_timesteps):
        if SAVE_GIFS:
            save_img = True if episode_number % SAVE_IMG_GAP == 0 else False
        else:
            save_img = False
        worker = TestWorker(
            self.meta_agent_id,
            self.local_network,
            episode_number,
            n_agent,
            fov,
            sensor_range,
            utility_range,
            budget_timesteps=budget_timesteps,
            device=self.device,
            save_image=save_img,
            greedy=GREEDY,
        )
        worker.run_episode()

        perf_metrics = worker.perf_metrics
        return perf_metrics

    def job(self, weights, episode_number, n_agent, fov, sensor_range, utility_range, budget_timesteps):
        # print("Starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_weights(weights)

        metrics = self.do_job(episode_number, n_agent, fov, sensor_range, utility_range, budget_timesteps)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return metrics, info


if __name__ == '__main__':
    start_time = time.time()
    ray.init()
    for i in range(NUM_RUN):
        run_test()
    print('Total time taken: {}'.format(time.time() - start_time))
