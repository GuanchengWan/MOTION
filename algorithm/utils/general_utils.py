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

import time


def coarsen_vector(x, C):
    return (C.power(2)).dot(x)

def lift_vector(x, C):
    # Pinv = C.T; Pinv[Pinv>0] = 1
    D = sp.sparse.diags(np.array(1 / np.sum(C, 0))[0])
    Pinv = (C.dot(D)).T
    return Pinv.dot(x)


def coarsen_matrix(W, C):
    # Pinv = C.T; #Pinv[Pinv>0] = 1
    D = sp.sparse.diags(np.array(1 / np.sum(C, 0))[0])
    Pinv = (C.dot(D)).T
    return (Pinv.T).dot(W.dot(Pinv))


def lift_matrix(W, C):
    P = C.power(2)
    return (P.T).dot(W.dot(P))


def get_coarsening_matrix(G, partitioning):
    # C = np.eye(G.N)
    degree = G.d
    C = sp.sparse.eye(G.N, format="lil")
    rows_to_delete = []
    for subgraph in partitioning:

        nc = len(subgraph)
        # print (subgraph)

        # add v_j's to v_i's row
        # print (degree[subgraph])
        total_degee = sum(degree[subgraph])
        # print (total_degee)
        # print (degree[subgraph]/total_degee)
        # quit()
        C[subgraph[0], subgraph] = np.sqrt(degree[subgraph]/total_degee)

        # C[subgraph[0], subgraph] = 1 / np.sqrt(nc)  # np.ones((1,nc))/np.sqrt(nc)
        # C[subgraph[0], subgraph[0]] = 1 / np.sqrt(nc)
        # C[subgraph[0], subgraph[1]] = 1 / np.sqrt(nc)
        # print (C[subgraph[0], subgraph])
        # quit()
        rows_to_delete.extend(subgraph[1:])

    # delete vertices
    # C = np.delete(C,rows_to_delete,0)

    C.rows = np.delete(C.rows, rows_to_delete)
    C.data = np.delete(C.data, rows_to_delete)
    C._shape = (G.N - len(rows_to_delete), G.N)

    C = sp.sparse.csc_matrix(C)

    # check that this is a projection matrix
    # assert sp.sparse.linalg.norm( ((C.T).dot(C))**2 - ((C.T).dot(C)) , ord='fro') < 1e-5

    return C


def coarsening_quality(G, C, kmax=30, Uk=None, lk=None):
    N = G.N
    I = np.eye(N)

    if (Uk is not None) and (lk is not None) and (len(lk) >= kmax):
        U, l = Uk, lk
    elif hasattr(G, "U"):
        U, l = G.U, G.e
    else:
        l, U = sp.sparse.linalg.eigsh(G.L, k=kmax, which="SM", tol=1e-3)

    l[0] = 1
    linv = l ** (-0.5)
    linv[0] = 0
    # l[0] = 0 # avoids divide by 0

    # below here, everything is C specific
    n = C.shape[0]
    Pi = C.T @ C
    S = graph_utils.get_S(G).T
    Lc = C.dot((G.L).dot(C.T))
    Lp = Pi @ G.L @ Pi

    if kmax > n / 2:
        [Uc, lc] = graph_utils.eig(Lc.toarray())
    else:
        lc, Uc = sp.sparse.linalg.eigsh(Lc, k=kmax, which="SM", tol=1e-3)

    if not sp.sparse.issparse(Lc):
        print("warning: Lc should be sparse.")

    metrics = {"r": 1 - n / N, "m": int((Lc.nnz - n) / 2)}

    kmax = np.clip(kmax, 1, n)

    # eigenvalue relative error
    metrics["error_eigenvalue"] = np.abs(l[:kmax] - lc[:kmax]) / l[:kmax]
    metrics["error_eigenvalue"][0] = 0

    # angles between eigenspaces
    metrics["angle_matrix"] = U.T @ C.T @ Uc

    # rss constants
    #    metrics['rss'] = np.diag(U.T @ Lp @ U)/l - 1
    #    metrics['rss'][0] = 0

    # other errors
    kmax = np.clip(kmax, 2, n)

    error_subspace = np.zeros(kmax)
    error_subspace_bound = np.zeros(kmax)
    error_sintheta = np.zeros(kmax)

    M = S @ Pi @ U @ np.diag(linv)
    #    M_bound = S @ (I - Pi) @ U @ np.diag(linv)

    for kIdx in range(1, kmax):
        error_subspace[kIdx] = np.abs(np.linalg.norm(M[:, : kIdx + 1], ord=2) - 1)
        #        error_subspace_bound[kIdx] = np.linalg.norm( M_bound[:,:kIdx + 1], ord=2)
        error_sintheta[kIdx] = (
            np.linalg.norm(metrics["angle_matrix"][0 : kIdx + 1, kIdx + 1 :], ord="fro")
            ** 2
        )

    metrics["error_subspace"] = error_subspace
    # metrics['error_subspace_bound'] = error_subspace_bound
    metrics["error_sintheta"] = error_sintheta

    return metrics


def plot_coarsening(
    Gall, Call, size=3, edge_width=0.8, node_size=20, alpha=0.55, title=""
):

    # colors signify the size of a coarsened subgraph ('k' is 1, 'g' is 2, 'b' is 3, and so on)
    colors = ["k", "g", "b", "r", "y"]

    n_levels = len(Gall) - 1
    if n_levels == 0:
        return None
    fig = plt.figure(figsize=(n_levels * size * 3, size * 2))

    for level in range(n_levels):

        G = Gall[level]
        edges = np.array(G.get_edge_list()[0:2])

        Gc = Gall[level + 1]
        #         Lc = C.dot(G.L.dot(C.T))
        #         Wc = sp.sparse.diags(Lc.diagonal(), 0) - Lc;
        #         Wc = (Wc + Wc.T) / 2
        #         Gc = gsp.graphs.Graph(Wc, coords=(C.power(2)).dot(G.coords))
        edges_c = np.array(Gc.get_edge_list()[0:2])
        C = Call[level]
        C = C.toarray()

        if G.coords.shape[1] == 2:
            ax = fig.add_subplot(1, n_levels + 1, level + 1)
            ax.axis("off")
            ax.set_title(f"{title} | level = {level}, N = {G.N}")

            [x, y] = G.coords.T
            for eIdx in range(0, edges.shape[1]):
                ax.plot(
                    x[edges[:, eIdx]],
                    y[edges[:, eIdx]],
                    color="k",
                    alpha=alpha,
                    lineWidth=edge_width,
                )
            for i in range(Gc.N):
                subgraph = np.arange(G.N)[C[i, :] > 0]
                ax.scatter(
                    x[subgraph],
                    y[subgraph],
                    c=colors[np.clip(len(subgraph) - 1, 0, 4)],
                    s=node_size * len(subgraph),
                    alpha=alpha,
                )

        elif G.coords.shape[1] == 3:
            ax = fig.add_subplot(1, n_levels + 1, level + 1, projection="3d")
            ax.axis("off")

            [x, y, z] = G.coords.T
            for eIdx in range(0, edges.shape[1]):
                ax.plot(
                    x[edges[:, eIdx]],
                    y[edges[:, eIdx]],
                    zs=z[edges[:, eIdx]],
                    color="k",
                    alpha=alpha,
                    lineWidth=edge_width,
                )
            for i in range(Gc.N):
                subgraph = np.arange(G.N)[C[i, :] > 0]
                ax.scatter(
                    x[subgraph],
                    y[subgraph],
                    z[subgraph],
                    c=colors[np.clip(len(subgraph) - 1, 0, 4)],
                    s=node_size * len(subgraph),
                    alpha=alpha,
                )

    # the final graph
    Gc = Gall[-1]
    edges_c = np.array(Gc.get_edge_list()[0:2])

    if G.coords.shape[1] == 2:
        ax = fig.add_subplot(1, n_levels + 1, n_levels + 1)
        ax.axis("off")
        [x, y] = Gc.coords.T
        ax.scatter(x, y, c="k", s=node_size, alpha=alpha)
        for eIdx in range(0, edges_c.shape[1]):
            ax.plot(
                x[edges_c[:, eIdx]],
                y[edges_c[:, eIdx]],
                color="k",
                alpha=alpha,
                lineWidth=edge_width,
            )

    elif G.coords.shape[1] == 3:
        ax = fig.add_subplot(1, n_levels + 1, n_levels + 1, projection="3d")
        ax.axis("off")
        [x, y, z] = Gc.coords.T
        ax.scatter(x, y, z, c="k", s=node_size, alpha=alpha)
        for eIdx in range(0, edges_c.shape[1]):
            ax.plot(
                x[edges_c[:, eIdx]],
                y[edges_c[:, eIdx]],
                z[edges_c[:, eIdx]],
                color="k",
                alpha=alpha,
                lineWidth=edge_width,
            )

    ax.set_title(f"{title} | level = {n_levels}, n = {Gc.N}")
    fig.tight_layout()
    return fig


################################################################################
# Variation-based contraction algorithms
################################################################################
def cosine_sim(x, y, eps=1e-08):
    div = np.linalg.norm(x)*np.linalg.norm(y)
    if abs(div) < eps:
        return np.dot(x, y)/eps
    return np.dot(x, y)/div

def coarsen_vector(x, C):
    return (C.power(2)).dot(x)

def coarsen_matrix(W, C):
    # Pinv = C.T; #Pinv[Pinv>0] = 1
    D = sp.sparse.diags(np.array(1 / np.sum(C, 0))[0])
    Pinv = (C.dot(D)).T
    return (Pinv.T).dot(W.dot(Pinv))