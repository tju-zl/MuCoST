import scanpy as sc
import torch
import numpy as np
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import to_undirected
from .utils import calculate_adj_matrix, search_l
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy


def preprocess_data(adata, arg):
    # highly variable gene
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

    # raw feature
    data = adata[:, adata.var['highly_variable']]
    adata.obsm['raw_feature'] = data.X.todense()

    # construct adjacency list
    spatial_edge = spatial_rknn(torch.FloatTensor(data.obsm['spatial']), arg).to('cuda')
    from sklearn.decomposition import PCA
    pca = PCA(n_components=arg.latent_dim, random_state=arg.seed)
    embedding = pca.fit_transform(data.X.toarray().copy())
    feature_edge = feature_knn(torch.FloatTensor(embedding).to('cuda'), arg)
    sf_edge = torch.concatenate([spatial_edge, feature_edge], dim=-1)

    # processing histology graph, revise from spagcn
    if arg.mode_his == 'his':
        img = adata.uns['image']
        x_pixel = adata.obsm["spatial"][:, 0]
        y_pixel = adata.obsm["spatial"][:, 1]
        s = 1
        b = 49
        adj = calculate_adj_matrix(x=x_pixel, y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s,
                                   histology=True)
        # print(adj)
        p = 0.5
        # Find the l value given p
        l = search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
        adj_exp = np.exp(-1 * (adj ** 2) / (2 * (l ** 2)))
        adj_exp[adj_exp < 0.01] = 0

        adj_matrix_sparse = scipy.sparse.coo_matrix(adj_exp)
        hist_wedge = from_scipy_sparse_matrix(adj_matrix_sparse)
        hist_adj = hist_wedge[0]

        print('Average spatial edge:', spatial_edge.size()[1] / data.n_obs)
        print('Average feature edge:', feature_edge.size()[1] / data.n_obs)
        print('Average histology edge:', hist_adj.size()[1] / data.n_obs)
        return spatial_edge, sf_edge, hist_wedge

    print('Average spatial edge:', spatial_edge.size()[1] / data.n_obs)
    print('Average feature edge:', feature_edge.size()[1] / data.n_obs)
    return spatial_edge, sf_edge



def spatial_rknn(data, arg):
    if arg.mode_rknn == 'rknn':
        edge_index = radius_graph(data, r=arg.radius, max_num_neighbors=6, flow=arg.flow)
        return to_undirected(edge_index)
    elif arg.mode_rknn == 'knn':
        edge_index = knn_graph(data, k=arg.rknn, flow=arg.flow)
        return to_undirected(edge_index)
    else:
        print('error: construction mode of spatial adjacency list must be set correctly.')



def feature_knn(data, arg):
    edge_index = knn_graph(data, k=arg.knn, flow=arg.flow, cosine=True)
    return edge_index






