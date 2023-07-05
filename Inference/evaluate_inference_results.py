import numpy as np
import torch
import torch.nn.functional as F

# Constants
MAX_RANK = 50


@torch.no_grad()
def compute_cosine_distance(features, others):
    """Computes cosine distance.
    Args:
        features (torch.Tensor): 2-D feature matrix.
        others (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    features = F.normalize(features, p=2, dim=1)
    others = F.normalize(others, p=2, dim=1)
    dist_m = 1 - torch.mm(features, others.t())
    return dist_m.cpu().numpy()


def evaluate(q_feat: torch.Tensor, g_features: torch.Tensor, q_pids: torch.Tensor, g_pids: torch.Tensor):
    dist_mat = compute_cosine_distance(q_feat, g_features)

    num_g = dist_mat.shape[1]
    if num_g < MAX_RANK:
        max_rank = num_g
        print('note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(dist_mat, axis=1)

    # compute cmc curve for each query
    all_cmc = []
    all_ap = []
    num_valid_q = 0.  # number of valid query

    for q_idx, q_pid in enumerate(q_pids):
        # compute cmc curve
        order = indices[q_idx]
        query_hit = (g_pids[order] == q_pid).astype(np.int32)
        if not np.any(query_hit):
            # this condition is true when query identity does not appear in gallery
            continue

        all_cmc.append(calc_cumulative_sum(query_hit)[:max_rank])
        all_ap.append(calc_average_precision(query_hit))
        num_valid_q += 1.

    assert num_valid_q > 0, 'error: all query identities do not appear in gallery'
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    return all_cmc, all_ap


def calc_cumulative_sum(query_hit):
    cmc = query_hit.cumsum()
    cmc[cmc > 1] = 1
    return cmc


def calc_average_precision(query_hit: np.ndarray):
    # compute average precision
    # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
    num_rel = query_hit.sum()
    tmp_cmc = query_hit.cumsum()
    tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
    tmp_cmc = np.asarray(tmp_cmc) * query_hit
    AP = tmp_cmc.sum() / num_rel
    return AP
