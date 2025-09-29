import numpy as np
import pygsp as gsp
from pygsp import graphs, filters, reduction
import scipy as sp
from scipy import sparse
import torch
import matplotlib
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from sortedcontainers import SortedList
import algorithm.utils.graph_utils as graph_utils
import algorithm.utils.general_utils as general_utils
import time
from scipy.sparse import diags

def coarsen(
    G_orig,
    G,
    replay_nodes,
    feature,
    K=10,
    r=0.5,
    max_levels=1,
    method="repro",
    Uk=None,
    lk=None,
    max_level_r=0.99,
):
    r = np.clip(r, 0, 0.999)
    G0 = G
    N = G.N

    # current and target graph sizes
    n, n_target = N, np.ceil((1 - r) * N)

    C = sp.sparse.eye(N, format="csc")
    Gc = G

    Call, Gall = [], []
    Gall.append(G_orig)

    for level in range(1, max_levels + 1):
        G = Gc
        r_cur = np.clip(1 - n_target / n, 0.0, max_level_r)

        if level == 1:
            if (Uk is not None) and (lk is not None) and (len(lk) >= K):
                mask = lk < 1e-10
                lk[mask] = 1
                lsinv = lk ** (-0.5)
                lsinv[mask] = 0
                B = Uk[:, :K] @ np.diag(lsinv[:K])
            else:
                offset = 2 * max(G.dw)
                T = offset * sp.sparse.eye(G.N, format="csc") - G.L
                lk, Uk = sp.sparse.linalg.eigsh(T, k=K, which="LM", tol=1e-5)
                lk = (offset - lk)[::-1]
                Uk = Uk[:, ::-1]
                mask = lk < 1e-10
                lk[mask] = 1
                lsinv = lk ** (-0.5)
                lsinv[mask] = 0
                B = Uk @ np.diag(lsinv)
            A = B
        else:
            B = iC.dot(B)
            d, V = np.linalg.eig(B.T @ (G.L).dot(B))
            mask = d == 0
            d[mask] = 1
            dinvsqrt = d ** (-1 / 2)
            dinvsqrt[mask] = 0
            A = B @ np.diag(dinvsqrt) @ V

        coarsening_list = repro(
            G, replay_nodes, feature, K=K, A=A, r=r_cur)

        iC = general_utils.get_coarsening_matrix(G, coarsening_list)

        C = iC.dot(C)
        Call.append(iC)

        Wc = graph_utils.zero_diag(general_utils.coarsen_matrix(G.W, iC))  # coarsen and remove self-loops
        Wc = (Wc + Wc.T) / 2  # this is only needed to avoid pygsp complaining for tiny errors

        if not hasattr(G, "coords"):
            Gc = gsp.graphs.Graph(Wc)
        else:
            Gc = gsp.graphs.Graph(Wc, coords=general_utils.coarsen_vector(G.coords, iC))
        
        
        Gc.W = graph_utils.zero_diag(Gc.W)

    Gc_ = G_orig
    for i in range(len(Call)):
        iC = Call[i]
        Wc = graph_utils.zero_diag(general_utils.coarsen_matrix(Gc_.W, iC))  # coarsen and remove self-loops
        Wc = (Wc + Wc.T) / 2  # this is only needed to avoid pygsp complaining for tiny errors

        if not hasattr(G, "coords"):
            Gc_ = gsp.graphs.Graph(Wc)
        else:
            Gc_ = gsp.graphs.Graph(Wc, coords=general_utils.coarsen_vector(G_orig.coords, iC))
        
        Gc_.W = graph_utils.zero_diag(Gc_.W)
        
        Gall.append(Gc_)

    return C, Gc_, Call, Gall

################################################################################
# Variation-based contraction algorithms
################################################################################
def cosine_sim(x, y, eps=1e-08):
    div = np.linalg.norm(x)*np.linalg.norm(y)
    if abs(div) < eps:
        return np.dot(x, y)/eps
    return np.dot(x, y)/div

def repro(G, replay_nodes, feature, A=None, K=10, r=0.5):
    # Proposed graph coarsening method: Node Representation Proximity
    N, deg, M = G.N, G.dw, G.Ne
    ones = np.ones(2)
    Pibot = np.eye(2) - np.outer(ones, ones) / 2

    # cost function for the edge
    def subgraph_cost(G, A, edge):
        edge, w = edge[:2].astype(int), edge[2]
        source, target = edge[0], edge[1]
        if source == target: return 0
        d = cosine_sim(feature[source].cpu().numpy(), feature[target].cpu().numpy())
        if edge[0] in replay_nodes or edge[1] in replay_nodes:
            return 100-d
        return -d

    edges = np.array(G.get_edge_list())
    weights = np.array([subgraph_cost(G, A, edges[:, e]) for e in range(M)])
    coarsening_list = matching(G, weights=-weights, r=r)

    return coarsening_list


def find(data, i):
    # print (data[i], i)
    if i != data[i]:
        data[i] = find(data, data[i])
    return data[i]

def union(data, i, j):
    # print (i,j)
    pi, pj = find(data, i), find(data, j)
    if pi != pj:
        data[pi] = pj
        return False
    return True

# def connected(data, i, j):
#     return find(data, i) == find(data, j)

def matching(G, weights, r):
    """
    Generates a matching greedily by selecting at each iteration the edge
    with the largest weight and then removing all adjacent edges from the
    candidate set.

    Parameters
    ----------
    G : pygsp graph
    weights : np.array(M)
        a weight for each edge
    r : float
        The desired dimensionality reduction (r = 1 - n/N)

    Notes:
    * The complexity of this is O(M)
    * Depending on G, the algorithm might fail to return ratios>0.3
    """

    N = G.N

    # the edge set
    edges = np.array(G.get_edge_list()[0:2])
    M = edges.shape[1]

    idx = np.argsort(-weights)
    # idx = np.argsort(weights)[::-1]
    edges = edges[:, idx]

    # the candidate edge set
    candidate_edges = edges.T.tolist()

    # the matching edge set (this is a list of arrays)
    matching = []

    # which vertices have been selected
    # marked = np.zeros(N, dtype=np.int32)
    marked = np.array([i for i in range(N)], dtype=np.int32)

    n, n_target = N, (1 - r) * N
    while len(candidate_edges) > 0:
        # pop a candidate edge
        [i, j] = candidate_edges.pop(0)
        # check if marked
        if i == j:
            continue
        if not union(marked, i, j):
            n -= 1
        # termination condition
        if n <= n_target:
            break

    matching = [[] for i in range(N)]
    for i in range(N):
        matching[find (marked, i)].append(i)
    matching = list(filter(lambda x: len(x) >= 2, matching))
    matching = [np.array(matching[i]) for i in range(len(matching))]
    return matching