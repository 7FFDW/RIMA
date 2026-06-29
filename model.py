import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd


class TPAMI_Config:
    wsi_feat_dim = 512
    num_genes = 4944
    num_anchors = 16
    gene_feat_dim = 512

    gw_epsilon = 0.05
    gw_iters = 2
    sinkhorn_iters = 5

    geo_dim = 64
    hidden_dim = 512

    learning_rate = 1e-4
    epochs = 2
    physical_batch_size = 1
    effective_batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = TPAMI_Config()


class GenomicPathwayEncoder(nn.Module):
    def __init__(self, csv_path=r'S:\code\TPAMI\gene\pathways_genes_matrix_new.csv', anchor_dim=None):
        super().__init__()


        pathway_df = pd.read_csv(csv_path, index_col=0)


        mask = torch.tensor(pathway_df.values, dtype=torch.float32)
        self.register_buffer('pathway_mask', mask)

        self.num_pathways = mask.shape[0]  # 186
        self.num_genes = mask.shape[1]  # 4944
        self.anchor_dim = anchor_dim


        self.weight = nn.Parameter(torch.Tensor(self.num_pathways, self.num_genes))
        self.bias = nn.Parameter(torch.Tensor(self.num_pathways))


        if anchor_dim > 1:
            self.feature_ext = nn.Sequential(
                nn.Linear(self.num_pathways, self.num_pathways * anchor_dim),
                nn.LayerNorm(self.num_pathways * anchor_dim),
                nn.GELU()
            )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):

        masked_weight = self.weight * self.pathway_mask


        pathway_scores = F.linear(x, masked_weight, self.bias)

        if self.anchor_dim > 1:

            out = self.feature_ext(pathway_scores)

            anchors = out.view(-1, self.num_pathways, self.anchor_dim)
        else:

            anchors = pathway_scores.unsqueeze(-1)


        return F.normalize(anchors, p=2, dim=-1)



class RiemannianMetricNet(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.feat_dim = feat_dim
        num_elements = (feat_dim * (feat_dim + 1)) // 2
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_elements)
        )

    def forward(self, x):
        B, N, D = x.shape
        avg_feat = torch.mean(x, dim=1)
        out = self.net(avg_feat)

        L = torch.zeros(B, D, D, device=x.device)
        tril_indices = torch.tril_indices(row=D, col=D, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = out

        d_idx = torch.arange(D)
        L[:, d_idx, d_idx] = torch.nn.functional.softplus(L[:, d_idx, d_idx]) + 1e-2

        G = torch.bmm(L, L.transpose(1, 2))
        G = G + 1e-4 * torch.eye(D, device=x.device).unsqueeze(0)
        return G



class RiemannianIntrinsicGWAlignment(nn.Module):
    def __init__(self, epsilon=0.05, gw_iters=3, sinkhorn_iters=5):
        super().__init__()
        self.epsilon = epsilon
        self.gw_iters = gw_iters
        self.sinkhorn_iters = sinkhorn_iters

    def compute_riemannian_cost(self, X, G):
        B, N, D = X.shape

        X_G = torch.bmm(X, G)  # [B, N, D]


        quad_X = torch.sum(X_G * X, dim=-1, keepdim=True)  # [B, N, 1]


        cross_term = torch.bmm(X_G, X.transpose(1, 2))  # [B, N, N]
        cost = quad_X + quad_X.transpose(1, 2) - 2 * cross_term

        cost = torch.clamp(cost, min=1e-6)
        return cost / (cost.max() + 1e-8)


    def sinkhorn(self, M, p, q):
        K_mat = torch.exp(-M / self.epsilon)
        u = torch.ones_like(p) / p.shape[1]
        for _ in range(self.sinkhorn_iters):
            v = q / (torch.bmm(K_mat.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
            u = p / (torch.bmm(K_mat, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
        return u.unsqueeze(-1) * K_mat * v.unsqueeze(1)

    def forward(self, wsi_features, gene_anchors, G_V, G_G):
        B, N, _ = wsi_features.shape
        _, K, _ = gene_anchors.shape
        C_V = self.compute_riemannian_cost(wsi_features, G_V)
        C_G = self.compute_riemannian_cost(gene_anchors, G_G)
        p = torch.ones(B, N, device=wsi_features.device) / N
        q = torch.ones(B, K, device=gene_anchors.device) / K
        T = torch.bmm(p.unsqueeze(-1), q.unsqueeze(1))
        C_V_sq, C_G_sq = C_V ** 2, C_G ** 2
        const_1 = torch.bmm(C_V_sq, p.unsqueeze(-1)).expand(-1, -1, K)
        const_2 = torch.bmm(q.unsqueeze(1), C_G_sq).expand(-1, N, -1)
        const_term = const_1 + const_2
        for _ in range(self.gw_iters):
            cross_term = torch.bmm(torch.bmm(C_V, T), C_G)
            M = const_term - 2 * cross_term
            T = self.sinkhorn(M, p, q)
        return T

class RiemannianGatedFusion(nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim_h * 2, dim_h), nn.Sigmoid())
        # self.mlp = nn.Sequential(nn.Linear(dim_h * 2, dim_h), nn.GELU())
    def forward(self, h_v, h_g):
        combined = torch.cat([h_v, h_g], dim=-1)
        g = self.gate(combined)
        fused = g * h_v + (1 - g) * h_g
        return torch.cat([fused, combined], dim=-1)


# ==========================================
# 3. 顶刊级数学正则计算
# ==========================================

def matrix_log(P, eps=1e-5):

    P = (P + P.transpose(-2, -1)) / 2.0


    I = torch.eye(P.size(-1), device=P.device, dtype=P.dtype)
    P = P + eps * I


    e, v = torch.linalg.eigh(P)


    log_e = torch.log(e)

    return torch.bmm(v, torch.bmm(torch.diag_embed(log_e), v.transpose(1, 2)))


def calculate_geometric_reg(T, G_V, G_G, v_geo, g_geo, lambda_metric=0.1, lambda_holo=0.01):

    log_G_V = matrix_log(G_V)
    log_G_G = matrix_log(G_G)
    loss_metric = torch.norm(log_G_V - log_G_G, p='fro')


    z_v = torch.complex(v_geo[:, :, 0], v_geo[:, :, 1])  # [1, N]
    z_g = torch.complex(g_geo[:, :, 0], g_geo[:, :, 1])  # [1, K]


    T_complex = T.to(torch.complex64)




    z_v_trans = torch.bmm(z_v.unsqueeze(1), T_complex).squeeze(1)


    loss_holo = torch.mean(torch.abs(z_v_trans - z_g) ** 2)



    return lambda_metric * loss_metric + lambda_holo * loss_holo


def cox_loss(risk_scores, survival_times, censor_statuses):
    risk_scores = risk_scores.view(-1)
    survival_times = survival_times.view(-1)
    censor_statuses = censor_statuses.view(-1)

    sorted_indices = torch.argsort(survival_times, descending=True)
    sorted_status = censor_statuses[sorted_indices]
    sorted_scores = risk_scores[sorted_indices]

    max_score = torch.max(sorted_scores)
    exp_scores = torch.exp(sorted_scores - max_score)
    risk_set_sum = torch.cumsum(exp_scores, dim=0)
    log_risk_set = torch.log(risk_set_sum + 1e-8) + max_score

    loss = -torch.sum((sorted_scores - log_risk_set) * sorted_status) / (torch.sum(sorted_status) + 1e-8)
    return loss




class RIMA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.v_proj = nn.Linear(cfg.wsi_feat_dim, cfg.geo_dim)
        self.g_proj = nn.Linear(cfg.gene_feat_dim, cfg.geo_dim)

        self.gene_encoder = GenomicPathwayEncoder(anchor_dim=cfg.gene_feat_dim)
        self.v_metric_net = RiemannianMetricNet(cfg.geo_dim, cfg.geo_dim // 2)
        self.g_metric_net = RiemannianMetricNet(cfg.geo_dim, cfg.geo_dim // 2)
        self.gw_aligner = RiemannianIntrinsicGWAlignment(cfg.gw_epsilon, cfg.gw_iters, cfg.sinkhorn_iters)

        self.lin_v = nn.Linear(cfg.wsi_feat_dim, cfg.hidden_dim)
        self.lin_g = nn.Linear(cfg.gene_feat_dim, cfg.hidden_dim)
        self.fusion = RiemannianGatedFusion(cfg.hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(cfg.hidden_dim * 3, 1))

        self.MLP = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

    def forward(self, wsi_features, gene_expr):


        if wsi_features.dim() == 4:
            wsi_features = wsi_features.squeeze(0)
        wsi_features = self.MLP(wsi_features)





        gene_anchors = self.gene_encoder(gene_expr)
        v_geo = self.v_proj(wsi_features)
        g_geo = self.g_proj(gene_anchors)

        G_V = self.v_metric_net(v_geo)
        G_G = self.g_metric_net(g_geo)
        T = self.gw_aligner(v_geo, g_geo, G_V, G_G)

        fused_v_points = torch.bmm(T.transpose(1, 2), wsi_features) * wsi_features.shape[1]
        h_v = torch.mean(fused_v_points, dim=1)
        h_g = torch.mean(gene_anchors, dim=1)

        h_v_proj = F.gelu(self.lin_v(h_v))
        h_g_proj = F.gelu(self.lin_g(h_g))

        z_fused = self.fusion(h_v_proj, h_g_proj)
        hazard = self.classifier(z_fused)

        return hazard, T, G_V, G_G, v_geo, g_geo




