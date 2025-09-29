import numpy as np
import pygsp as gsp
import scipy.sparse as sp

def to_networkx():
    import networkx as nx
    return nx.from_scipy_sparse_matrix(G.W)
    
def get_neighbors(G, i):
    return G.A[i,:].indices
    # return np.arange(G.N)[np.array((G.W[i,:] > 0).todense())[0]]
    
def get_giant_component(G):

    from scipy.sparse import csgraph

    [ncomp, labels] = csgraph.connected_components(G.W, directed=False, return_labels=True)

    W_g = np.array((0,0))
    coords_g = np.array((0,2))
    keep = np.array(0)
    
    for i in range(0,ncomp):
        
        idx = np.where(labels!=i)
        idx = idx[0]
        
        if G.N-len(idx) > W_g.shape[0]:        
            W_g = G.W.toarray()
            W_g = np.delete(W_g, idx, axis=0)
            W_g = np.delete(W_g, idx, axis=1)
            if hasattr(G, 'coords'):
                coords_g = np.delete(G.coords, idx, axis=0)
            keep = np.delete(np.arange(G.N), idx)    

    np.fill_diagonal(W_g, 0)

    if not hasattr(G, 'coords'):
        # print(W_g.shape)
        G_g = gsp.graphs.Graph(W=W_g)        
    else:
        G_g = gsp.graphs.Graph(W=W_g, coords=coords_g)
    
    G_g.W = zero_diag(G_g.W)
    
    return (G_g, keep)


def get_S(G):
    """
    Construct the N x |E| gradient matrix S
    """
    # the edge set
    edges = G.get_edge_list()
    weights = np.array(edges[2])
    edges = np.array(edges[0:2])
    M = edges.shape[1]
    
    # Construct the N x |E| gradient matrix S
    S = np.zeros((G.N,M))
    for e in np.arange(M):
        S[edges[0,e], e] = np.sqrt(weights[e])
        S[edges[1,e], e] = -np.sqrt(weights[e])
        
    return S   

# Compare the spectum of L and Lc
def eig(A, order='ascend'):

    # eigenvalue decomposition
    [l,X] = np.linalg.eigh(A)

    # reordering indices     
    idx = l.argsort()   
    if order == 'descend':
        idx = idx[::-1]

    # reordering     
    l = np.real(l[idx])
    X = X[:, idx]
    return (X,np.real(l))

def zero_diag(W):

    if isinstance(W, np.ndarray):
        W_out = W.copy()
        np.fill_diagonal(W_out, 0)
        return W_out
    else:

        W_lil = W.tolil()
        for i in range(W.shape[0]):
            W_lil[i, i] = 0
        return W_lil.tocsr()

def is_symmetric(As):

    from scipy import sparse 
    
    if As.shape[0] != As.shape[1]:
        return False

    if not isinstance(As, sparse.coo_matrix):
        As = sparse.coo_matrix(As)

    r, c, v = As.row, As.col, As.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check

def create_graph_with_zero_diag(W, coords=None):


    W_zero_diag = zero_diag(W)
    

    if coords is not None:
        G = gsp.graphs.Graph(W_zero_diag, coords=coords)
    else:
        G = gsp.graphs.Graph(W_zero_diag)
    

    G.W = zero_diag(G.W)
    
    G.compute_laplacian()
    G.compute_fourier_basis()
    
    return G
