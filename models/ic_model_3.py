import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_data import b_FPS, k_points
from utils.utils_model import loss_fn, ProjectMLP, momentum_update
from utils.attention import CrossAttnBlock
from utils.encoders import DGCNN_CLS_Encoder_1
from models.ic_model_1 import get_patch_idx, get_features, get_1_branch_feats


class SimAttention_3(nn.Module):
    """PoCCA with new trick, online and target branches not merging"""
    def __init__(self, patch_num):
        super(SimAttention_3, self).__init__()
        self.online_encoder = DGCNN_CLS_Encoder_1().cuda()
        self.online_projector = ProjectMLP().cuda()
        self.online_attn = CrossAttnBlock().cuda()
        self.predictor = ProjectMLP().cuda()
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
        gf_2, pf_2 = get_1_branch_feats(aug2, self.patch_num, self.target_encoder)
        gf_3, pf_3 = get_1_branch_feats(aug2, self.patch_num, self.online_encoder)
        gf_4, pf_4 = get_1_branch_feats(aug1, self.patch_num, self.target_encoder)

        # get loss
        loss_1 = loss_fn(self.predictor(self.online_projector(self.online_attn(gf_1, pf_1))),
                         self.target_projector(self.target_attn(gf_2, pf_2)))
        loss_2 = loss_fn(self.predictor(self.online_projector(self.online_attn(gf_3, pf_3))),
                         self.target_projector(self.target_attn(gf_4, pf_4)))
        loss = loss_1 + loss_2
        return loss.mean()
      
