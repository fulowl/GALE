import torch
import torch.nn.functional as F
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def loss_net(anchor, pagerank_values, feat, orbits, use_pr=False):
    
    discriminator=lambda x, y: x @ y.t()
    
    if not use_pr:
        pr_diff = torch.abs(orbits.unsqueeze(1) - orbits.unsqueeze(0)).to('cuda')  # (N, N)
        pos_mask = (pr_diff < 1e-12).float()
    else:
        pr_diff = torch.abs(pagerank_values.unsqueeze(1) - pagerank_values.unsqueeze(0)).to('cuda')  # (N, N)
        pos_mask = (pr_diff < 1e-12).float()

    feat_ = feat.detach().cpu().numpy()
    dist_matrix = euclidean_distances(feat_, feat_)
    dist_mask = (torch.tensor(dist_matrix) < 1e-8).float().to('cuda')
    
    pos_mask = pos_mask * dist_mask
    
    neg_mask = 1. - pos_mask
    
    num_neg = neg_mask.int().sum()
    num_pos = pos_mask.int().sum()
    similarity = discriminator(anchor, anchor)

    E_pos = (np.log(2) - F.softplus(- similarity * pos_mask)).sum()
    
    E_pos /= num_pos

    neg_sim = similarity * neg_mask

    E_neg = (F.softplus(neg_sim) - np.log(2)).sum()
    
    E_neg /= num_neg

    return E_neg - E_pos

def batch_loss(embeddings, part_values, feat, batch):
    
    batch_mask = (batch.unsqueeze(1) == batch.unsqueeze(0))
    
    feat_ = feat.detach().cpu().numpy()
    dist_matrix = euclidean_distances(feat_, feat_)
    dist_mask = (torch.tensor(dist_matrix) < 1e-12).float().to('cuda')
    
    part_diff = torch.abs(part_values.unsqueeze(1) - part_values.unsqueeze(0)).to('cuda')  # (N, N)

    pos_mask = (part_diff < 1e-12).float()
    
    pos_mask = pos_mask * dist_mask

    neg_mask = 1. - pos_mask
    
    num_neg = neg_mask.int().sum()
    num_pos = pos_mask.int().sum()
    
    discriminator=lambda x, y: x @ y.t()
    similarity = discriminator(embeddings, embeddings)

    E_pos = (np.log(2) - F.softplus(- similarity * pos_mask * batch_mask)).sum()
    E_pos /= num_pos

    neg_sim = similarity * neg_mask * batch_mask
    E_neg = (F.softplus(neg_sim) - np.log(2)).sum()
    E_neg /= num_neg

    return E_neg - E_pos