#!/usr/bin/env python3

import os
import pickle

import numpy as np
import networkx as nx

import pyAgrum as gum
import pyAgrum.lib.image as gumimage
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph

import utils as u

GROUND_TRUTH = 'ground-truth'
VERBOSE = True

SEED = 2
NODES = 10
MIN_DEGREE = 1
MAX_DEGREE = 3
ANOMALOUS_NODES = 1
NORMAL_SAMPLES = 1_000
ANOMALOUS_SAMPLES = 1_000
STATES = 6

SRC_DIR_TEMPLATE = 'data/s-{SEED}/n-{NODES}-d-{DEGREE}-an-{ANOMALOUS_NODES}-nor-s-{NORMAL_SAMPLES}-an-s-{ANOMALOUS_SAMPLES}/'


def draw_and_save(bn, target, samples, nodes):
    generator = gum.BNDatabaseGenerator(bn)
    generator.drawSamples(samples)

    var_order = [u.get_node_name(node) for node in range(nodes)]
    generator.setVarOrder(var_order)
    generator.toCSV(target)

def generate_random_dag(n, max_degree):
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)

    perm = np.random.permutation(n)
    current_i = 1
    while current_i < n:
        r = int(np.random.uniform(low=MIN_DEGREE, high=max_degree + 1))
        parents = perm[current_i: current_i + r]
        for p in parents:
            G.add_edge(perm[current_i - 1], p)
        current_i += 1

    if not nx.is_directed_acyclic_graph(G):
        print(f"Warning: Created DAG is not acyclic!")

    return G.edges()

def get_random_dag(n, max_degree=3, states=STATES):
    edges = generate_random_dag(n, max_degree)

    bn = gum.BayesNet('BN')
    g_graph = GeneralGraph([])
    for node in range(n):
        _node = u.get_node_name(node)
        g_graph.add_node(GraphNode(_node))
        bn.add(gum.RangeVariable(_node, str(node), 0, states - 1))

    for e in edges:
        bn.addArc(u.get_node_name(e[0]), u.get_node_name(e[1]))
        g_graph.add_directed_edge(GraphNode(u.get_node_name(e[0])),
                                  GraphNode(u.get_node_name(e[1])))

    bn.generateCPTs()
    return bn, g_graph

def inject_failure(bn, a_nodes):
    for node in a_nodes:
        # Change the distribution of the anomalous node
        bn.generateCPT(node)

def generate_data(seed, nodes, max_degree, normal_samples, anomalous_samples, anomalous_nodes, states, verbose=VERBOSE):
    src_dir = SRC_DIR_TEMPLATE.format(
        SEED=seed,
        NODES=nodes,
        DEGREE=max_degree,
        NORMAL_SAMPLES=normal_samples,
        ANOMALOUS_NODES=anomalous_nodes,
        ANOMALOUS_SAMPLES=anomalous_samples)
    if not os.path.exists(src_dir):
        try: os.makedirs(src_dir)
        except: pass

    np.random.seed(seed)
    gum.initRandom(seed)

    an_nodes = [u.get_node_name(x) for x in np.random.choice(nodes, anomalous_nodes, replace=False)]
    if verbose:
        print(f"Randomly assigned anomalous node(s) {an_nodes}")

    # Create a random DAG
    bn, g_graph = get_random_dag(nodes, max_degree=max_degree, states=states)

    gumimage.export(bn, src_dir + GROUND_TRUTH + '.pdf',
                    nodeColor={n: 0 for n in an_nodes})

    with open(src_dir + 'g_graph.pkl', 'wb') as f:
        pickle.dump(g_graph, f)

    draw_and_save(bn, src_dir + 'normal.csv', normal_samples, nodes)
    inject_failure(bn, an_nodes)
    draw_and_save(bn, src_dir + 'anomalous.csv', anomalous_samples, nodes)

    if verbose:
        print(f"Data is saved at {src_dir}")

    # Choose a front-end service.
    # Only used for some baselines
    fe_service = an_nodes[0]
    while True:
        succ = [u.get_node_name(n) for n in bn.children(fe_service)]
        if len(succ) == 0: break
        fe_service = np.random.choice(succ)

    return src_dir, fe_service, an_nodes

if __name__ == '__main__':
    generate_data(SEED, NODES, MAX_DEGREE, NORMAL_SAMPLES, ANOMALOUS_SAMPLES, ANOMALOUS_NODES, STATES, VERBOSE)
