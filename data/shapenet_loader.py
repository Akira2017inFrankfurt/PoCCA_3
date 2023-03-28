import os
import numpy as np
from torch.utils.data import Dataset
from pointwolf import PointWOLF


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


if __name__ == "__main__":
    import torch
    print('start doing sth!')
    # test_root = r'/root/autodl-nas/'
    test_root = r'/Users/huangqianliang/PycharmProjects/pythonProject/data'
    t_dataset = ShapeNetCLS(root=test_root, npoints=1024)
    print('Total item number is: ', t_dataset.__len__())  # 57448
    train_loader = torch.utils.data.DataLoader(t_dataset, batch_size=2, shuffle=True)
    test_pn_1, test_pn_2, original = 0, 0, 0
    for m1, m2 in train_loader:
        print(m1.shape)  # [B, 1024, 3]
        print('///')
        print(m2.shape)  # [B, 1024, 3]
        test_pn_1, test_pn_2 = m1, m2
        break

    # # test for visualization
    # from utils.visualize import visualization
    #
    # def vis(test_pn):
    #     test_pn = test_pn.squeeze()
    #     print(test_pn.shape)
    #
    #     test_pn = test_pn.cpu().numpy()
    #     point_set = test_pn.astype(np.float32)[:, 0:3]
    #     visualization(point_set)

    # vis(original)
    # vis(test_pn_1)
    # vis(test_pn_2)