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
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=arg.hvg_n)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

    # raw feature
    data = adata[:, adata.var['highly_variable']].copy()
    adata.obsm['raw_feature'] = data.X.toarray()

    # construct adjacency list
    spatial_edge = spatial_rknn(torch.FloatTensor(adata.obsm['spatial']), arg).to('cuda')

    from sklearn.decomposition import PCA
    pca = PCA(n_components=arg.latent_dim, random_state=arg.seed)
    embedding = pca.fit_transform(data.X.toarray().copy())
    feature_edge = feature_knn(torch.FloatTensor(embedding).to('cuda'), arg)

    # # refine spatial_edge
    # s_edge_ind = spatial_edge.clone()
    # spatial_edge = remove_edges_by_condition(s_edge_ind, 0., embedding)
    # refine feature_edge

    if arg.cut_corr:
        edge_ind = feature_edge.clone()
        feature_edge = remove_edges_by_condition(edge_ind, arg.corr, embedding)

    sf_edge = torch.concatenate([spatial_edge, feature_edge], dim=-1)

    # GCN中对spatial graph和 featural graph 分别给与不同的权重
    s_e_attr = arg.e_attr * torch.ones((spatial_edge.size()[1]), dtype=torch.float).to('cuda')
    f_e_attr = (1 - arg.e_attr) * torch.ones((feature_edge.size()[1]), dtype=torch.float).to('cuda')
    sf_e_attr = torch.cat((s_e_attr, f_e_attr))

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

        print('Average spatial edge:', spatial_edge.size()[1] / adata.n_obs)
        print('Average feature edge:', feature_edge.size()[1] / adata.n_obs)
        print('Average histology edge:', hist_adj.size()[1] / adata.n_obs)
        return adata, spatial_edge, sf_edge, hist_wedge

    print('Average spatial edge:', spatial_edge.size()[1] / adata.n_obs)
    print('Average feature edge:', feature_edge.size()[1] / adata.n_obs)
    return adata, spatial_edge, sf_edge, s_e_attr, sf_e_attr
    # return adata, sf_edge, sf_edge, s_e_attr, sf_e_attr



def spatial_rknn(data, arg):
    if arg.mode_rknn == 'rknn':
        edge_index = radius_graph(data, r=arg.radius, max_num_neighbors=arg.rknn, flow=arg.flow)
        return to_undirected(edge_index)
    elif arg.mode_rknn == 'knn':
        edge_index = knn_graph(data, k=arg.rknn, flow=arg.flow)
        return to_undirected(edge_index)
    else:
        print('error: construction mode of spatial adjacency list must be set correctly.')


def feature_knn(data, arg):
    edge_index = knn_graph(data, k=arg.knn, flow=arg.flow, cosine=True)
    return edge_index


def condition(edge_index, embedding, c):
    corr_index = []
    for i in range(edge_index.size(1)):
        # if cosine_similarity(embedding[edge_index[0,i]], embedding[edge_index[1,i]]) <= 0.5:
        if pearson_sim(embedding[edge_index[0, i]], embedding[edge_index[1, i]]) <= c:
            corr_index.append(True)
        else:
            corr_index.append(False)
    return corr_index


def remove_edges_by_condition(edge_index, c, emb):
    mask = condition(edge_index, emb, c)  # 创建布尔掩码
    filtered_edge_index = edge_index[:, ~torch.tensor(mask)]  # 应用掩码以删除满足条件的边
    return filtered_edge_index


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity


def pearson_sim(v1, v2):
    corr, _ = pearsonr(v1, v2)
    return corr


from scipy.stats import kendalltau


def kendalltau_sim(v1, v2):
    corr, p_value = kendalltau(v1, v2)
    return corr


from scipy.stats import spearmanr


def spearmanr_sim(v1, v2):
    corr, p_value = spearmanr(v1, v2)
    return corr


from copulas.multivariate.gaussian import GaussianMultivariate


def copulas_sim(v1, v2):
    copula = GaussianMultivariate()
    data = [v1, v2]
    copula.fit(data)
    corr = copula.squared_correlation
    return corr


# Pearson Correlation
from scipy.stats import pearsonr


def pearson_knn(data, arg):
    corr_mat = torch.zeros(data.shape[1], data.shape[1])
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            corr, _ = pearsonr(data[:, i].numpu(), data[:, j].numpy())
            corr_mat[i, j] = corr
            corr_mat[j, i] = corr


# Copula Entropy
def ce_knn(data, arg):
    pass


# kendall correlation
def kendall_knn(data, arg):
    pass


# Spearman 's rank coefficient of correlation
def spearman_knn(data, arg):
    pass
