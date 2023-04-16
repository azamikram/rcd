#!/usr/bin/env python3

import os
import pickle
from argparse import ArgumentParser

import numpy as np
import networkx as nx

# pyAgrum is for graphical network
import pyAgrum as gum
import pyAgrum.lib.image as gum_image
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--node-size", type=int, default=10)
parser.add_argument("--min-degree", type=int, default=1)
parser.add_argument("--max-degree", type=int, default=3)
parser.add_argument("--anomalous-node-size", type=int, default=1)
parser.add_argument("--normal-sample-size", type=int, default=1000)
parser.add_argument("--anomalous-sample-size", type=int, default=1000)
parser.add_argument("--states", type=int, default=6)
args = parser.parse_args()


def x(i: int) -> str:
    return f"X{i}"


def draw_and_save(bn, target, samples, node_size):
    """
    Generate samples from the given BN and save them to a CSV file.

    Parameters
    ----------
    bn : pyAgrum.BayesNet
        The BN to generate samples from.
    target : str
        The path to save the generated samples.
    samples : int
        The number of samples to generate.
    node_size: int
        The number of nodes in the BN.
    """
    generator = gum.BNDatabaseGenerator(bn)
    generator.drawSamples(samples)

    var_order = [x(i) for i in range(node_size)]
    generator.setVarOrder(var_order)
    generator.toCSV(target)


def get_random_dag(n, min_degree, max_degree, states):
    """
    Generate a random DAG with `n` nodes and `max_degree`
    as the maximum degree of each node.

    Returns:
    --------
    bayes_net: pyAgrum.BayesNet
    general_graph: causallearn.graph.GeneralGraph
    """
    # TODO: describe the process of create a DAG 
    # there are involvements of three libraries? seems redundant to me.

    # ============= PART 1: GENERATE EDGES ========================
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    # generate a random permutation of the nodes
    perm_node_ids = np.random.permutation(n)

    # for each node, random select parents in the subsequent nodes
    # assign links from left (child) to right (parents)
    cur_id = 0
    while cur_id < n - 1:
        par_size = np.random.randint(min_degree, max_degree + 1)
        parents = perm_node_ids[cur_id + 1 : cur_id + 1 + par_size]

        G.add_edges_from([(perm_node_ids[cur_id], p) for p in parents])
        cur_id += 1

    if not nx.is_directed_acyclic_graph(G):
        print(f"Warning: Created DAG is not acyclic!")

    edges = G.edges()


    # ============= PART 2: GENERATE BAYES NET ========================
    bayes_net = gum.BayesNet("BN")
    general_graph = GeneralGraph([])
    for i in range(n):
        general_graph.add_node(GraphNode(x(i)))
        bayes_net.add(gum.RangeVariable(x(i), str(i), 0, states - 1))

    for e in edges:
        general_graph.add_directed_edge(GraphNode(x(e[0])), GraphNode(x(e[1])))
        bayes_net.addArc(x(e[0]), x(e[1]))

    # # generate random conditional probability table for the given structure
    bayes_net.generateCPTs()
    return bayes_net, general_graph

# def inject_failure(bn, a_node_size):
#     # change the distribution of the anomalous node
#     for node in a_node_size:
#         bn.generateCPT(node)


def main():
    # parse arguments
    seed = args.seed
    node_size = args.node_size
    min_degree = args.min_degree
    max_degree = args.max_degree
    norm_size = args.normal_sample_size
    anom_size = args.anomalous_sample_size
    anom_node_size = args.anomalous_node_size
    states = args.states

    # prepare data dir for saving the generated data
    data_dir = f"data/s-{seed}/n-{node_size}-d-{max_degree}-an-{anom_node_size}-nor-s-{norm_size}-an-s-{anom_size}/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # random seed
    np.random.seed(seed)
    gum.initRandom(seed)

    # choose anomalous node ids
    anom_node_ids = [x(i) for i in np.random.choice(node_size, anom_node_size)]
    print(f"Randomly assigned anomalous node(s) {anom_node_ids}")

    # Create a random DAG
    bayes_net, general_graph = get_random_dag(n=node_size, min_degree=min_degree, max_degree=max_degree, states=states)

    gum_image.export(
        bayes_net, data_dir + "ground-truth.pdf", nodeColor={n: 0 for n in anom_node_ids}
    )

    # with open(data_dir + "g_graph.pkl", "wb") as f:
    #     pickle.dump(g_graph, f)

    # draw_and_save(bn, data_dir + "normal.csv", normal_sample_size, node_size)
    # inject_failure(bn, anom_node_ids)
    # draw_and_save(bn, data_dir + "anomalous.csv", anomalous_sample_size, node_size)

    # print(f"Data is saved at {data_dir}")

    # # Choose a front-end service.
    # # Only used for some baselines
    # fe_service = anom_node_ids[0]
    # while True:
    #     succ = [x(i) for i in bn.children(fe_service)]
    #     if len(succ) == 0:
    #         break
    #     fe_service = np.random.choice(succ)

    # return data_dir, fe_service, anom_node_ids


if __name__ == "__main__":
    main()
