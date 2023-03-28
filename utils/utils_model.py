import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_data import new_k_patch


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


@torch.no_grad()
def momentum_update(online, target, tao=0.99):
    if target is None:
        target = copy.deepcopy(online)
    else:
        for online_params, target_params in zip(online.parameters(), target.parameters()):
            target_weight, online_weight = target_params.data, online_params.data
            target_params.data = target_weight * tao + (1 - tao) * online_weight
    for parameter in target.parameters():
        parameter.requires_grad = False
    return target


# get patch features
def get_patch_feature(patches, encoder):
    num_patches = patches.shape[1]
    current_patch = patches[:, 0, :, :].squeeze()
    patch_features = encoder(current_patch)
    for i in range(1, num_patches):
        current_patch = patches[:, i, :, :].squeeze()
        current_patch_feat = encoder(current_patch)
        patch_features = torch.cat((patch_features, current_patch_feat), dim=1)
    return patch_features


def concat_features(patch_fs_list):
    feature = patch_fs_list[0]
    for i in range(1, len(patch_fs_list)):
        feature = torch.cat((feature, patch_fs_list[i]), dim=1)
    return feature


def get_2_branch_patch_features(data1, data2, patch_number_list, encoder):
    patches1 = [new_k_patch(data1, scale) for scale in patch_number_list]
    patches2 = [new_k_patch(data2, scale) for scale in patch_number_list]
    patch_feats1 = [get_patch_feature(patch, encoder) for patch in patches1]
    patch_feats2 = [get_patch_feature(patch, encoder) for patch in patches2]
    patch_feats1 = concat_features(patch_feats1)
    patch_feats2 = concat_features(patch_feats2)
    return concat_features([patch_feats1, patch_feats2])


# projector and predictor network
class ProjectMLP(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, hidden_size=4096):
        super(ProjectMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(input_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.bn(self.l1(x.reshape(x.shape[0], -1)))
        x = self.l2(self.relu(x))
        return x.reshape(x.shape[0], 1, -1)