#!/usr/bin/env python3

import os, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import rcd
import gen_data as data
import utils as u

THREADING = True
WORKERS = 16 - 2
RESULT_DIR = 'exp_results'
VERBOSE = False

K = 1
SEED_MAX = 10_000
MAX_DEGREE = 3
ANOMALOUS_NODES = 1
STATES = 6

SEED = 42
EXECUTIONS = 100
NODES = [10, 20, 40, 80, 160]
NORMAL_SAMPLES = 10_000
ANOMALOUS_SAMPLES = 10_000

def run_single_experiment(seed, nodes):
    # Generate data
    (src_dir, _, a_nodes) = data.generate_data(seed, nodes, MAX_DEGREE,
                                               NORMAL_SAMPLES, ANOMALOUS_SAMPLES,
                                               ANOMALOUS_NODES, STATES, VERBOSE)

    def _basic_extract_result(result, prefix):
        return {
            f"{prefix}_tests": result['tests'],
            f"{prefix}_time": round(result['time'], 3),
            f"{prefix}_detected_rc": result['root_cause'],
        }

    def _extract_result(result, prefix):
        rank = 0
        k_rc = result['root_cause'][:K]
        for x in k_rc:
            if x in a_nodes:
                rank += 1

        recall = rank / len(a_nodes)
        precision = 0 if len(result['root_cause']) == 0 else rank / len(result['root_cause'])
        return {
            **_basic_extract_result({**result}, prefix),
            f"{prefix}_recall": recall,
            f"{prefix}_top_k_targets": k_rc,
            f"{prefix}_precision": precision,
            f"{prefix}_f1_score": 0 if precision == 0 and recall == 0 else 2 * ((precision * recall) / (precision + recall))
        }

    (normal_df, anomalous_df) = u.load_datasets(src_dir + 'normal.csv',
                                                src_dir + 'anomalous.csv')

    rcd_r = rcd.top_k_rc(normal_df, anomalous_df, K,
                         None, seed=seed, localized=True, verbose=VERBOSE)
    rcd_r = _extract_result(rcd_r, 'rcd')

    result_list = []
    result = {'nodes': nodes, 'seed': seed, 'a_nodes': a_nodes}
    result_list.append({**result, **rcd_r})
    if VERBOSE:
        print(f"Output: {result}")
    return pd.DataFrame(result_list, columns=result_list[0].keys())

def run_executions(n, nodes=0, seed=None, file_path=None, header=True):
    if THREADING:
        t_pool = Pool(WORKERS)
        future = [None] * n

    local_header = header
    df = pd.DataFrame()
    def _df_append(result):
        nonlocal local_header
        if file_path:
            df = pd.DataFrame(result)
            df.to_csv(file_path, index=False, mode='a', header=local_header)
            local_header = False
        else:
            df = df.append(result)

    for i in range(n):
        if VERBOSE:
            print(f"Running for nodes {nodes} for execution {i}")
        _seed = int(SEED_MAX * np.random.uniform()) if seed is None else seed
        if THREADING:
            future[i] = t_pool.starmap_async(run_single_experiment, [(_seed, nodes)])
        else:
            print(f'seed={_seed}')
            _df_append(run_single_experiment(_seed, nodes))

    if THREADING:
        for i in range(n):
            _df_append(future[i].get()[0])
        t_pool.close()
        t_pool.join()
    return df

# Change the total number of nodes
def different_nodes(nodes, seed, execs):
    np.random.seed(seed)
    now = int(time.time())
    dir = f"{RESULT_DIR}/{now}/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    msg = f"""
    Different nodes experiments
    Testing time, accuracy, and number of CI tests with different nodes
    seed = {seed}
    nodes = {nodes}
    execs = {execs}
    MAX_DEGREE = {MAX_DEGREE}
        """
    readme = open(dir + 'readme.txt', 'w')
    readme.write(msg)
    readme.close()

    for i, node in enumerate(nodes):
        print(f"Running the experiment with {node} nodes")
        run_executions(execs, node, file_path=dir + 'data.csv', header=(i == 0))
    return dir

if __name__ == '__main__':
    dir = different_nodes(NODES, SEED, EXECUTIONS)
    print(f"The result of the experiment is stored at {dir}")
