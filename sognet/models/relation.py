#!/usr/bin/python
# @Author  : LaoYang
# @Email   : lhy_ustb@pku.edu.cn
# @Software: VsCode

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

# ----------------------------------------------------------------------------------- #
#  Relation Head
# ----------------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationHead(nn.Module):

    def __init__(self):
        super(RelationHead, self).__init__()
        self.U = nn.Sequential(
                nn.Linear(80, 256, bias=False),
                nn.ReLU(inplace=True),
                )
        self.V = nn.Sequential(
                nn.Linear(80, 256, bias=False),
                nn.ReLU(inplace=True)
                )
        self.W = nn.Sequential(
                nn.Conv2d(4, 128, 1, bias=False),
                nn.ReLU(inplace=True),
                )
        self.P = nn.Sequential(
                nn.Conv2d(384, 1, 1, bias=False),
                # nn.ReLU(inplace=True),
                )
        nn.init.kaiming_normal_(self.P[0].weight)

    def extract_position_matrix(self, rois):
        xmin = rois[:, 1]
        ymin = rois[:, 2]
        xmax = rois[:, 3]
        ymax = rois[:, 4]
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        x_ctr = xmin + 0.5 * w
        y_ctr = ymin + 0.5 * h

        dx = -(x_ctr[:, None] - x_ctr)
        dx = dx / w[:, None]

        dy = -(y_ctr[:, None] - y_ctr)
        dy = dy / h[:, None]

        dw = torch.log(w / w[:, None])
        dh = torch.log(h / h[:, None])

        return torch.cat([dx[..., None], dy[..., None], dw[..., None], dh[..., None]], dim=2)

    def cls2onehot(self, cls_idx):
        assert (cls_idx >= 1).all()
        assert (cls_idx <= 80).all()
        cls_idx = cls_idx - 1
        onehot = torch.arange(80).to(cls_idx.device)
        onehot = (cls_idx[:, None] == onehot).float()
        return onehot

    def cls_relation(self, onehot):
        feat1 = self.U(onehot)
        feat2 = self.V(onehot)
        cls_relation = feat1[:, None, :] * feat2
        return cls_relation

    def forward(self, mask, bbox=None, cls_idx=None):
        if len(cls_idx) <= 1:
            return mask, torch.tensor([0]).to(mask.device)
        onehot = self.cls2onehot(cls_idx)
        cls_relation = self.cls_relation(onehot)

        bbox_mat = self.extract_position_matrix(bbox)
        bbox_feat = self.W(bbox_mat.permute(2, 0, 1)[None, ...]).squeeze(0).permute(1, 2, 0)

        feat = torch.cat([cls_relation, bbox_feat], dim=2)
        logit = self.P(feat.permute(2, 0, 1)[None, ...])
        R = torch.sigmoid(logit).squeeze(0).squeeze(0)
        OO = R - R.transpose(0, 1)
        O = F.relu(OO, inplace=True)

        # post process
        trans_mask = torch.sigmoid(mask)
        overlap_part = (mask * trans_mask)[:, :, None, ...] * trans_mask
        overlap_part = overlap_part * O[..., None, None]
        overlap_part = overlap_part.sum(dim=2)
        new_mask = mask - overlap_part
        return new_mask, O

class RelationLoss(nn.Module):

    def __init__(self):
        super(RelationLoss, self).__init__()

    def forward(self, relation_mat, gt, new=False):
        num_ins = relation_mat.size(0)
        if num_ins <= 1:
            return torch.tensor(0.).to(relation_mat.device)

        if not new:
            relation_mat = relation_mat + relation_mat.transpose(0, 1)


        loss = F.mse_loss(relation_mat, gt)
        if False and new and relation_mat.device == torch.device("cuda:0"):
            print('\n')

            format_str = '{:>4} : {:>3}/{:>3}, {:.4f}'
            try:
                one_item = relation_mat[gt > 0.5]
                zero_item = relation_mat[gt < 0.5]
                one_true_num = (one_item > 0.5).sum().item()
                zero_true_num = (zero_item < 0.5).sum().item()
                print(format_str.format('one', one_true_num, one_item.size(0), one_true_num * 1.0 / one_item.size(0)))
                print(one_item)
                print(format_str.format('zero', zero_true_num, zero_item.size(0), zero_true_num * 1.0 / zero_item.size(0)))
                print(zero_item)
            except ZeroDivisionError:
                print('Zero is divided!')
        return loss
