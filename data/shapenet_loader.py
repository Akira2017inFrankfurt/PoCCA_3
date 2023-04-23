import os
import numpy as np
from torch.utils.data import Dataset
from data.pointwolf import PointWOLF
from data.rstj_aug import rstj_aug


def points_sampler(points, num):
    pt_idxs = np.arange(0, points.shape[0])
    np.random.shuffle(pt_idxs)
    points = points[pt_idxs[0:num], :]
    return points


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ShapeNetCLS(Dataset):
    # output original point cloud and m1, m2
    def __init__(self, root, npoints):
        self.npoints = npoints
        self.root = root
        self.train_npy = os.path.join(self.root, "shapenet57448xyzonly.npz")
        self.td = dict(np.load(self.train_npy))
        self.data = self.td["data"]
        self.aug1 = PointWOLF(0.4)
        self.aug2 = PointWOLF(0.8)

    def __getitem__(self, index):
        point_set = self.data[index]
        _, morph_1 = self.aug1(pc_normalize(point_set))
        _, morph_2 = self.aug2(pc_normalize(point_set))

        morph_1 = points_sampler(morph_1, self.npoints)
        morph_2 = points_sampler(morph_2, self.npoints)

        return morph_1, morph_2

    def __len__(self):
        return len(self.data)

    
class ShapeNetCLS_RSTJ(Dataset):
    def __init__(self, root, npoints):
        self.npoints = npoints
        self.root = root
        self.train_npy = os.path.join(self.root, "shapenet57448xyzonly.npz")
        self.td = dict(np.load(self.train_npy))
        self.data = self.td["data"]
        self.aug1 = rstj_aug
        self.aug2 = rstj_aug

    def __getitem__(self, index):
        point_set = self.data[index]
        morph_1 = self.aug1(pc_normalize(point_set))
        morph_2 = self.aug2(pc_normalize(point_set))

        morph_1 = points_sampler(morph_1, self.npoints)
        morph_2 = points_sampler(morph_2, self.npoints)

        return morph_1, morph_2

    def __len__(self):
        return len(self.data)
    
