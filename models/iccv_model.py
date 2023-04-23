import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_data import b_FPS, k_points
from utils.utils_model import loss_fn, ProjectMLP, momentum_update
from utils.attention import CrossAttnBlock
from utils.encoders import DGCNN_CLS_Encoder_1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_patch_idx(x, patch_num, points_number=None):
    if points_number is None:
        points_number = int(2048 / patch_num)
    _, patch_center_xyz = b_FPS(x, patch_num)
    idx = k_points(patch_center_xyz, x, points_number)
    idx = idx.permute(0, 2, 1)
    return idx


def get_features(points_features, patch_idx):
    # input: points_features [bs, 1024, 2048]
    # input: patch_idx [bs, 256, 8]
    # output: patch_features [bs, 1024, 256, 8] -> [bs, 1024, 8]
    bs, dim, _ = points_features.size()
    num_patch = patch_idx.shape[-1]
    # global feature
    global_feature = F.adaptive_max_pool1d(points_features, 1).view(bs, 1, dim)

    # [bs, 1024, 2048] -> [bs, 1024, 2048, 1]
    # [bs, 256, 8] -> [bs, 1, 256, 8]
    # broadcast and gather(points_features[:, :, :, None], 2, patch_ids[:, None, :, :])
    # bs, 1024, 256, 8
    points_features = points_features[:, :, :, None].expand(-1, -1, -1, patch_idx.size(-1))
    patch_idx = patch_idx[:, None, :, :].expand(-1, points_features.size(1), -1, -1)
    result = torch.gather(points_features, 2, patch_idx)
    # get feature via max_pooling
    # bs, 1024, 256, 8 -> bs*8, 1024, 256
    result = result.permute(0, 3, 1, 2)
    # print('result shape: ', result.shape)
    result = result.reshape(-1, dim, result.size(-1))
    # print('result shape: ', result.shape)
    # not sure can direct do it
    patch_features = F.adaptive_max_pool1d(result, 1)
    patch_features = patch_features.permute(0, 2, 1)
    patch_features = patch_features.reshape(bs, num_patch, dim)
    # print('patch features: ', patch_features.shape)

    return global_feature, patch_features


def get_1_branch_feats(morph, patch_num, encoder):
    # _, g_idx = b_FPS(morph, 1024)
    p_idx = get_patch_idx(morph, patch_num)
    point_features = encoder(morph)
    g_feature, p_features = get_features(point_features, p_idx)
    return g_feature, p_features


class SimAttention_1(nn.Module):
    """PoCCA with new trick"""

    def __init__(self, patch_num):
        super(SimAttention_ICCV_1, self).__init__()
        self.online_encoder = DGCNN_CLS_Encoder_1().to(device)
        self.online_projector = ProjectMLP().to(device)
        self.online_attn = CrossAttnBlock().to(device)
        self.predictor = ProjectMLP().to(device)
        self.patch_num = patch_num
        self.target_encoder = None
        self.target_projector = None
        self.target_attn = None

    def forward(self, aug1, aug2):
        self.target_encoder = momentum_update(self.online_encoder, self.target_encoder)
        self.target_projector = momentum_update(self.online_projector, self.target_projector)
        self.target_attn = momentum_update(self.online_attn, self.target_attn)

        # get g and p features
        gf_1, pf_1 = get_1_branch_feats(aug1, self.patch_num, self.online_encoder)
        _, pf_2 = get_1_branch_feats(aug2, self.patch_num, self.online_encoder)
        gf_2, _ = get_1_branch_feats(aug2, self.patch_num, self.target_encoder)

        gf_3, pf_3 = get_1_branch_feats(aug2, self.patch_num, self.online_encoder)
        _, pf_4 = get_1_branch_feats(aug1, self.patch_num, self.online_encoder)
        gf_4, _ = get_1_branch_feats(aug1, self.patch_num, self.target_encoder)

        # concat features
        con_pf_1 = torch.cat((pf_1, pf_2), dim=1)
        con_pf_2 = torch.cat((pf_3, pf_4), dim=1)

        # get loss
        loss_1 = loss_fn(self.predictor(self.online_projector(self.online_attn(gf_1, con_pf_1))),
                         self.target_projector(self.target_attn(gf_2, con_pf_1)))
        loss_2 = loss_fn(self.predictor(self.online_projector(self.online_attn(gf_3, con_pf_2))),
                         self.target_projector(self.target_attn(gf_4, con_pf_2)))
        loss = loss_1 + loss_2
        return loss.mean()


if __name__ == "__main__":
    import torch
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rand_x_0 = torch.rand([4, 2048, 3]).to(device)
    rand_x_1 = torch.rand([4, 2048, 3]).to(device)

    test_model = SimAttention_1(patch_num=4)

    rand_out = test_model(rand_x_0, rand_x_1)
    print('Done! output is: ', rand_out)
