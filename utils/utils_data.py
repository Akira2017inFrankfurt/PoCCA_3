import math
import torch
import random
import numpy as np


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1))).cuda()
    return res.reshape(*raw_size, -1)


def b_FPS(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).cuda()
    distance = torch.ones(B, N).cuda() * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).cuda()
    batch_indices = torch.arange(B, dtype=torch.long).cuda()
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1).cuda()
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids, index_points(xyz, centroids)


def k_points(a, b, k):
    # a: small, b: big one
    inner = -2 * torch.matmul(a, b.transpose(2, 1))
    aa = torch.sum(a ** 2, dim=2, keepdim=True)
    bb = torch.sum(b ** 2, dim=2, keepdim=True)
    pairwise_distance = -aa - inner - bb.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def new_k_patch(x, scale):
    """
    - scale: 0, 1, 2  pow(2, scale) ---> 1, 2, 4
    - num of neighbor points ---> 256, 512, 1024
    - then all downsample to 256 points
    """
    n_patch = 8
    n_points = int(2048 / n_patch)
    patch_centers_index, center_point_xyz = b_FPS(x, n_patch)
    idx = k_points(center_point_xyz, x, int(n_points * math.pow(2, scale)))
    idx = idx.permute(0, 2, 1)
    new_patch = torch.zeros([n_patch, x.shape[0], n_points, x.shape[-1]]).cuda()
    for i in range(n_patch):
        patch_idx = idx[:, :, i].reshape(x.shape[0], -1)
        _, patch_points = b_FPS(index_points(x, patch_idx), n_points)
        new_patch[i] = patch_points
    new_patch = new_patch.permute(1, 0, 2, 3) - center_point_xyz.unsqueeze(2)

    return new_patch
