import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_data import b_FPS, k_points
from utils.utils_model import loss_fn, ProjectMLP, momentum_update, PosEncoding
from utils.attention import CrossAttnBlock
from utils.encoders import DGCNN_CLS_Encoder_1
from models.ic_model_1 import get_patch_idx, get_features, get_1_branch_feats


class SimAttention_2(nn.Module):
    """PoCCA with new trick, and positional encoding"""
    def __init__(self, patch_num):
        super(SimAttention_2, self).__init__()
        self.online_encoder = DGCNN_CLS_Encoder_1().cuda()
        self.online_projector = ProjectMLP().cuda()
        self.online_attn = CrossAttnBlock().cuda()
        self.predictor = ProjectMLP().cuda()
        self.pos_en = PosEncoding().cuda()

        self.after_add_pos_en = nn.Linear(1088, 1024).cuda()

        self.patch_num = patch_num
        self.target_encoder = None
        self.target_projector = None
        self.target_attn = None

    def forward(self, aug1, aug2):
        self.target_encoder = momentum_update(self.online_encoder, self.target_encoder)
        self.target_projector = momentum_update(self.online_projector, self.target_projector)
        self.target_attn = momentum_update(self.online_attn, self.target_attn)

        # get g and p features, and patch_centroids
        gf_1, pf_1, patch_centroids_1 = get_1_branch_feats(aug1, self.patch_num, self.online_encoder)
        _, pf_2, patch_centroids_2 = get_1_branch_feats(aug2, self.patch_num, self.online_encoder)
        gf_2, _, _ = get_1_branch_feats(aug2, self.patch_num, self.target_encoder)
        # print('pf_1: ', pf_1.shape)  # bs, num_patch, dim
        # print('patch_centroids_1: ', patch_centroids_1.shape)  # bs, num_patch, 3
        # print('pos_en_1: ', self.pos_en(patch_centroids_1).shape)  # bs, num_patch, 64
        pf_1 = self.after_add_pos_en(torch.cat((pf_1, self.pos_en(patch_centroids_1)), dim=-1))
        pf_2 = self.after_add_pos_en(torch.cat((pf_2, self.pos_en(patch_centroids_2)), dim=-1))
        
        gf_3, pf_3, patch_centroids_3 = get_1_branch_feats(aug2, self.patch_num, self.online_encoder)
        _, pf_4, patch_centroids_4 = get_1_branch_feats(aug1, self.patch_num, self.online_encoder)
        gf_4, _, _ = get_1_branch_feats(aug1, self.patch_num, self.target_encoder)
        pf_3 = self.after_add_pos_en(torch.cat((pf_3, self.pos_en(patch_centroids_3)), dim=-1))
        pf_4 = self.after_add_pos_en(torch.cat((pf_4, self.pos_en(patch_centroids_4)), dim=-1))

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
