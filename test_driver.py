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

os.environ.setdefault('MARVEL_CONFIG_MODE', 'test')

from utils.model import PolicyNet
from utils.test_worker import TestWorker
from utils.runtime_config import *
import csv

EVAL_USE_GPU = USE_GPU and NUM_GPU > 0 and torch.cuda.is_available()

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
    all_n_agent = [2, 4, 8]
    all_sensor_range = [8, 10, 15]
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
                        ]

                        with open("record.txt", "a") as f:
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
