from copy import deepcopy

from tqdm.auto import tqdm

from causallearn.graph.Edges import Edges
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.ChoiceGenerator import ChoiceGenerator
from causallearn.utils.cit import *
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

citest_cache = dict()


def possible_parents(node_x, adjx, knowledge=None):
    possibleParents = []

    for node_z in adjx:
        if (knowledge is None) or (
            not knowledge.is_forbidden(node_z, node_x)
            and not knowledge.is_required(node_x, node_z)
        ):
            possibleParents.append(node_z)

    return possibleParents


def freeDegree(nodes, adjacencies):
    max = 0
    for node_x in nodes:
        opposites = adjacencies[node_x]
        for node_y in opposites:
            adjx = set(opposites)
            adjx.remove(node_y)

            if len(adjx) > max:
                max = len(adjx)
    return max


def forbiddenEdge(node_x, node_y, knowledge):
    if knowledge is None:
        return False
    elif knowledge.is_forbidden(node_x, node_y) and knowledge.is_forbidden(
        node_y, node_x
    ):
        print(
            node_x.get_name()
            + " --- "
            + node_y.get_name()
            + " because it was forbidden by background background_knowledge."
        )
        return True
    return False


def searchAtDepth0(
    data,
    nodes,
    adjacencies,
    sep_sets,
    independence_test_method=fisherz,
    alpha=0.05,
    verbose=False,
    knowledge=None,
    pbar=None,
    cache_variables_map=None,
):
    empty = []
    if cache_variables_map is None:
        data_hash_key = hash(str(data))
        ci_test_hash_key = hash(independence_test_method)
        if independence_test_method == chisq or independence_test_method == gsq:
            cardinalities = np.max(data, axis=0) + 1
        else:
            cardinalities = None
    else:
        data_hash_key = cache_variables_map["data_hash_key"]
        ci_test_hash_key = cache_variables_map["ci_test_hash_key"]
        cardinalities = cache_variables_map["cardinalities"]

    show_progress = not pbar is None
    if show_progress:
        pbar.reset()
    for i in range(len(nodes)):
        if show_progress:
            pbar.update()
        if show_progress:
            pbar.set_description(f"Depth=0, working on node {i}")
        if verbose and (i + 1) % 100 == 0:
            print(nodes[i + 1].get_name())

        for j in range(i + 1, len(nodes)):
            ijS_key = (i, j, frozenset(), data_hash_key, ci_test_hash_key)
            if ijS_key in citest_cache:
                p_value = citest_cache[ijS_key]
            else:
                p_value = (
                    independence_test_method(data, i, j, tuple(empty))
                    if cardinalities is None
                    else independence_test_method(
                        data, i, j, tuple(empty), cardinalities
                    )
                )
                citest_cache[ijS_key] = p_value
            independent = p_value > alpha

            if verbose:
                print(
                    nodes[i].get_name()
                    + f" {'ind' if independent else 'dep'} "
                    + nodes[j].get_name()
                    + " | (), p-value = "
                    + str(p_value)
                )

            no_edge_required = (
                True
                if knowledge is None
                else (
                    (not knowledge.is_required(nodes[i], nodes[j]))
                    or knowledge.is_required(nodes[j], nodes[i])
                )
            )
            if independent and no_edge_required:
                sep_sets[(i, j)] = set()

            elif not forbiddenEdge(nodes[i], nodes[j], knowledge):
                adjacencies[nodes[i]].add(nodes[j])
                adjacencies[nodes[j]].add(nodes[i])
    if show_progress:
        pbar.refresh()
    return freeDegree(nodes, adjacencies) > 0


def searchAtDepth(
    data,
    depth,
    nodes,
    adjacencies,
    sep_sets,
    independence_test_method=fisherz,
    alpha=0.05,
    verbose=False,
    knowledge=None,
    pbar=None,
    cache_variables_map=None,
):
    def edge(adjx, i, adjacencies_completed_edge):
        for j in range(len(adjx)):
            node_y = adjx[j]
            _adjx = list(adjacencies_completed_edge[nodes[i]])
            _adjx.remove(node_y)
            ppx = possible_parents(nodes[i], _adjx, knowledge)

            if len(ppx) >= depth:
                cg = ChoiceGenerator(len(ppx), depth)
                choice = cg.next()
                flag = False
                while choice is not None:
                    cond_set = [nodes.index(ppx[index]) for index in choice]
                    choice = cg.next()

                    Y = nodes.index(adjx[j])
                    X, Y = (i, Y) if (i < Y) else (Y, i)
                    XYS_key = (
                        X,
                        Y,
                        frozenset(cond_set),
                        data_hash_key,
                        ci_test_hash_key,
                    )
                    if XYS_key in citest_cache:
                        p_value = citest_cache[XYS_key]
                    else:
                        p_value = (
                            independence_test_method(data, X, Y, tuple(cond_set))
                            if cardinalities is None
                            else independence_test_method(
                                data, X, Y, tuple(cond_set), cardinalities
                            )
                        )

                        citest_cache[XYS_key] = p_value

                    independent = p_value > alpha
                    if independent:
                        if verbose:
                            message = f"{nodes[i].get_name()} {'ind' if independent else 'dep'} {adjx[j].get_name()} | ("
                            for cond_set_index in range(len(cond_set)):
                                message += nodes[cond_set[cond_set_index]].get_name()
                                if cond_set_index != len(cond_set) - 1:
                                    message += ", "
                            message += "), p-value = " + str(p_value)
                            print(message)

                    no_edge_required = (
                        True
                        if knowledge is None
                        else (
                            (not knowledge.is_required(nodes[i], adjx[j]))
                            or knowledge.is_required(adjx[j], nodes[i])
                        )
                    )
                    if independent and no_edge_required:
                        if adjacencies[nodes[i]].__contains__(adjx[j]):
                            adjacencies[nodes[i]].remove(adjx[j])
                        if adjacencies[adjx[j]].__contains__(nodes[i]):
                            adjacencies[adjx[j]].remove(nodes[i])

                        if cond_set is not None:
                            if sep_sets.keys().__contains__((i, nodes.index(adjx[j]))):
                                sep_set = sep_sets[(i, nodes.index(adjx[j]))]
                                for cond_set_item in cond_set:
                                    sep_set.add(cond_set_item)
                            else:
                                sep_sets[(i, nodes.index(adjx[j]))] = set(cond_set)

                        flag = True
                if flag:
                    return False
        return True

    if cache_variables_map is None:
        data_hash_key = hash(str(data))
        ci_test_hash_key = hash(independence_test_method)
        if independence_test_method == chisq or independence_test_method == gsq:
            cardinalities = np.max(data, axis=0) + 1
        else:
            cardinalities = None
    else:
        data_hash_key = cache_variables_map["data_hash_key"]
        ci_test_hash_key = cache_variables_map["ci_test_hash_key"]
        cardinalities = cache_variables_map["cardinalities"]

    count = 0

    adjacencies_completed = deepcopy(adjacencies)

    show_progress = not pbar is None
    if show_progress:
        pbar.reset()

    for i in range(len(nodes)):
        if show_progress:
            pbar.update()
        if show_progress:
            pbar.set_description(f"Depth={depth}, working on node {i}")
        if verbose:
            count += 1
            if count % 10 == 0:
                print("count " + str(count) + " of " + str(len(nodes)))
        adjx = list(adjacencies[nodes[i]])
        finish_flag = False
        while not finish_flag:
            finish_flag = edge(adjx, i, adjacencies_completed)

            adjx = list(adjacencies[nodes[i]])
    if show_progress:
        pbar.refresh()
    return freeDegree(nodes, adjacencies) > depth


def searchAtDepth_not_stable(
    data,
    depth,
    nodes,
    adjacencies,
    sep_sets,
    independence_test_method=fisherz,
    alpha=0.05,
    verbose=False,
    knowledge=None,
    pbar=None,
    cache_variables_map=None,
):
    def edge(adjx, i, adjacencies_completed_edge):
        for j in range(len(adjx)):
            node_y = adjx[j]
            _adjx = list(adjacencies_completed_edge[nodes[i]])
            _adjx.remove(node_y)
            ppx = possible_parents(nodes[i], _adjx, knowledge)

            if len(ppx) >= depth:
                cg = ChoiceGenerator(len(ppx), depth)
                choice = cg.next()

                while choice is not None:
                    cond_set = [nodes.index(ppx[index]) for index in choice]
                    choice = cg.next()

                    Y = nodes.index(adjx[j])
                    X, Y = (i, Y) if (i < Y) else (Y, i)
                    XYS_key = (
                        X,
                        Y,
                        frozenset(cond_set),
                        data_hash_key,
                        ci_test_hash_key,
                    )
                    if XYS_key in citest_cache:
                        p_value = citest_cache[XYS_key]
                    else:
                        p_value = (
                            independence_test_method(data, X, Y, tuple(cond_set))
                            if cardinalities is None
                            else independence_test_method(
                                data, X, Y, tuple(cond_set), cardinalities
                            )
                        )

                        citest_cache[XYS_key] = p_value

                    independent = p_value > alpha

                    no_edge_required = (
                        True
                        if knowledge is None
                        else (
                            (not knowledge.is_required(nodes[i], adjx[j]))
                            or knowledge.is_required(adjx[j], nodes[i])
                        )
                    )
                    if independent and no_edge_required:
                        if adjacencies[nodes[i]].__contains__(adjx[j]):
                            adjacencies[nodes[i]].remove(adjx[j])
                        if adjacencies[adjx[j]].__contains__(nodes[i]):
                            adjacencies[adjx[j]].remove(nodes[i])

                        if cond_set is not None:
                            if sep_sets.keys().__contains__((i, nodes.index(adjx[j]))):
                                sep_set = sep_sets[(i, nodes.index(adjx[j]))]
                                for cond_set_item in cond_set:
                                    sep_set.add(cond_set_item)
                            else:
                                sep_sets[(i, nodes.index(adjx[j]))] = set(cond_set)

                        if verbose:
                            message = (
                                "Independence accepted: "
                                + nodes[i].get_name()
                                + " _||_ "
                                + adjx[j].get_name()
                                + " | "
                            )
                            for cond_set_index in range(len(cond_set)):
                                message += nodes[cond_set[cond_set_index]].get_name()
                                if cond_set_index != len(cond_set) - 1:
                                    message += ", "
                            message += "\tp = " + str(p_value)
                            print(message)
                        return False
        return True

    if cache_variables_map is None:
        data_hash_key = hash(str(data))
        ci_test_hash_key = hash(independence_test_method)
        if independence_test_method == chisq or independence_test_method == gsq:
            cardinalities = np.max(data, axis=0) + 1
        else:
            cardinalities = None
    else:
        data_hash_key = cache_variables_map["data_hash_key"]
        ci_test_hash_key = cache_variables_map["ci_test_hash_key"]
        cardinalities = cache_variables_map["cardinalities"]

    count = 0

    show_progress = not pbar is None
    if show_progress:
        pbar.reset()

    for i in range(len(nodes)):
        if show_progress:
            pbar.update()
        if show_progress:
            pbar.set_description(f"Depth={depth}, working on node {i}")
        if verbose:
            count += 1
            if count % 10 == 0:
                print("count " + str(count) + " of " + str(len(nodes)))
        adjx = list(adjacencies[nodes[i]])
        finish_flag = False
        while not finish_flag:
            finish_flag = edge(adjx, i, adjacencies)

            adjx = list(adjacencies[nodes[i]])
    if show_progress:
        pbar.refresh()
    return freeDegree(nodes, adjacencies) > depth


def fas(
    data,
    nodes,
    independence_test_method=fisherz,
    alpha=0.05,
    knowledge=None,
    depth=-1,
    verbose=False,
    stable=True,
    show_progress=True,
    cache_variables_map=None,
):
    """
    Implements the "fast adjacency search" used in several causal algorithm in this file. In the fast adjacency
    search, at a given stage of the search, an edge X*-*Y is removed from the graph if X _||_ Y | S, where S is a subset
    of size d either of adj(X) or of adj(Y), where d is the depth of the search. The fast adjacency search performs this
    procedure for each pair of adjacent edges in the graph and for each depth d = 0, 1, 2, ..., d1, where d1 is either
    the maximum depth or else the first such depth at which no edges can be removed. The interpretation of this adjacency
    search is different for different algorithm, depending on the assumptions of the algorithm. A mapping from {x, y} to
    S({x, y}) is returned for edges x *-* y that have been removed.

    Parameters
    ----------
    data: data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    nodes: The search nodes.
    independence_test_method: the function of the independence test being used
            [fisherz, chisq, gsq, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - kci: Kernel-based conditional independence test
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    knowledge: background background_knowledge
    depth: the depth for the fast adjacency search, or -1 if unlimited
    verbose: True is verbose output should be printed or logged
    stable: run stabilized skeleton discovery if True (default = True)
    show_progress: whether to use tqdm to show progress bar
    cache_variables_map: This variable a map which contains the variables relate with cache. If it is not None,
                            it should contain 'data_hash_key' 、'ci_test_hash_key' and 'cardinalities'.

    Returns
    -------
    graph: Causal graph skeleton, where graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j.
    sep_sets: separated sets of graph
    """

    # --------check parameter -----------
    if (depth is not None) and type(depth) != int:
        raise TypeError("'depth' must be 'int' type!")
    if (knowledge is not None) and type(knowledge) != BackgroundKnowledge:
        raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")

    # --------end check parameter -----------

    # ------- initial variable -----------
    sep_sets = {}
    adjacencies = {node: set() for node in nodes}
    if depth is None or depth < 0:
        depth = 1000

    def _unique(column):
        return np.unique(column, return_inverse=True)[1]

    if independence_test_method == chisq or independence_test_method == gsq:
        data = np.apply_along_axis(_unique, 0, data).astype(np.int64)

    if cache_variables_map is None:
        if independence_test_method == chisq or independence_test_method == gsq:
            cardinalities = np.max(data, axis=0) + 1
        else:
            cardinalities = None
        cache_variables_map = {
            "data_hash_key": hash(str(data)),
            "ci_test_hash_key": hash(independence_test_method),
            "cardinalities": cardinalities,
        }
    # ------- end initial variable ---------
    if verbose:
        print("Starting Fast Adjacency Search.")

    # use tqdm to show progress bar
    pbar = tqdm(total=len(nodes)) if show_progress else None
    for d in range(depth):
        more = False

        if d == 0:
            more = searchAtDepth0(
                data,
                nodes,
                adjacencies,
                sep_sets,
                independence_test_method,
                alpha,
                verbose,
                knowledge,
                pbar=pbar,
                cache_variables_map=cache_variables_map,
            )
        else:
            if stable:
                more = searchAtDepth(
                    data,
                    d,
                    nodes,
                    adjacencies,
                    sep_sets,
                    independence_test_method,
                    alpha,
                    verbose,
                    knowledge,
                    pbar=pbar,
                    cache_variables_map=cache_variables_map,
                )
            else:
                more = searchAtDepth_not_stable(
                    data,
                    d,
                    nodes,
                    adjacencies,
                    sep_sets,
                    independence_test_method,
                    alpha,
                    verbose,
                    knowledge,
                    pbar=pbar,
                    cache_variables_map=cache_variables_map,
                )
        if not more:
            break
    if show_progress:
        pbar.close()

    graph = GeneralGraph(nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_x = nodes[i]
            node_y = nodes[j]
            if adjacencies[node_x].__contains__(node_y):
                graph.add_edge(Edges().undirected_edge(node_x, node_y))

    if verbose:
        print("Finishing Fast Adjacency Search.")
    return graph, sep_sets
