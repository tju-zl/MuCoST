import torch
import tqdm.notebook as tq
from .model import Model
from .utils import fix_seed
from .data import preprocess_data


def training_model(adata, arg):
    fix_seed(seed=arg.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if arg.mode_his == 'his':
        spatial_adj, feature_adj, histology_ew = preprocess_data(adata, arg)
        raw_feature = torch.FloatTensor(adata.obsm['raw_feature'].copy()).to(device)

        g_h, w_h = histology_ew
        sh_edge = torch.concatenate([spatial_adj, g_h.to(device)], dim=-1)
        s_h = torch.ones(spatial_adj.size()[1]).to(device)
        sh_weight = torch.concatenate([s_h, w_h.to(device)], dim=-1)

        model = Model(raw_feature.shape[1], arg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)

        for ep in tq.tqdm(range(1, arg.epoch+1)):
            model.train()
            optimizer.zero_grad()
            loss = model(raw_feature, spatial_adj, feature_adj, sh_edge, sh_weight)[-1]

            if ep % (arg.epoch/arg.log_step) == 0:
                print(f'EP[%4d]: loss=%.4f.' % (ep, loss.data))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            latent_rep = model(raw_feature, spatial_adj, feature_adj, sh_edge, sh_weight)[0]
        latent = latent_rep.to('cpu').detach().numpy()
        adata.obsm['MuCoST'] = latent

    elif arg.mode_his == 'noh':
        spatial_adj, feature_adj = preprocess_data(adata, arg)
        raw_feature = torch.FloatTensor(adata.obsm['raw_feature'].copy()).to(device)

        model = Model(raw_feature.shape[1], arg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
        losses = []
        for ep in tq.tqdm(range(1, arg.epoch + 1)):
            model.train()
            optimizer.zero_grad()
            loss = model(raw_feature, spatial_adj, feature_adj)[-1]
            losses.append(loss.item())

            if ep % (arg.epoch / arg.log_step) == 0:
                print(f'EP[%4d]: loss=%.4f.' % (ep, loss.data))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        import matplotlib.pyplot as plt
        x = range(1, len(losses) + 1)
        plt.plot(x, losses)
        plt.show()

        model.eval()
        with torch.no_grad():
            latent_rep = model(raw_feature, spatial_adj, feature_adj)[0]
        latent = latent_rep.to('cpu').detach().numpy()
        adata.obsm['MuCoST'] = latent
