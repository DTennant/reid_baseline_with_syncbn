from .eval_reid import eval_func
from .re_ranking import re_ranking
import torch


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def re_rank(q, g):
    qq_dist = euclidean_dist(q, q).numpy()
    gg_dist = euclidean_dist(g, g).numpy()
    qg_dist = euclidean_dist(q, g).numpy()
    distmat = re_ranking(qg_dist, qq_dist, gg_dist)
    return distmat
