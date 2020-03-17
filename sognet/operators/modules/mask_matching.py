import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
import matplotlib
import matplotlib.pyplot
from sognet.config.config import config
from sognet.bbox.bbox_transform import bbox_transform, bbox_overlaps
import cv2

class MaskMatching(nn.Module):

    def __init__(self, num_seg_classes, enable_void, class_mapping=None):
        super(MaskMatching, self).__init__()
        self.class_mapping = dict(zip(range(1, config.dataset.num_classes), range(num_seg_classes - config.dataset.num_classes + 1, num_seg_classes))) if class_mapping is None else class_mapping
        self.num_seg_classes = num_seg_classes
        self.num_inst_classes = len(self.class_mapping)
        self.enable_void = enable_void

    @staticmethod
    def get_relation_gt(gt_masks, cls_idx=None):
        num_ins = gt_masks.size(0)
        gt_masks = (gt_masks == 1).float()
        intersection = (gt_masks[:, None, :, :] * gt_masks).view(num_ins, num_ins, -1)
        intersection_num = intersection.sum(dim=2)
        mask_num = gt_masks.view(num_ins, -1).sum(dim=1)
        min_area = torch.min(mask_num[:, None], mask_num)
        intersection_ratio = intersection_num.float() / min_area.float()
        intersection_ratio -= torch.eye(num_ins).float().cuda()
        relation_mat = (intersection_ratio >= 0.1).float()
        # mask = (cls_idx[:, None] != cls_idx).float()
        # return relation_mat * mask
        return relation_mat

    def forward(self, gt_segs, gt_masks, keep_inds=None, pan_gt=None, sample=False, cls_idx=None):
        """
        :param gt_segs: [1 x h x w]
        :param gt_masks: [num_gt_boxes x h x w]
        :param keep_inds: [num_kept_boxes x 1]
        :return: matched_gt: [1 x h x w]
        """

        matched_gt = torch.ones_like(gt_segs) * -1
        matched_gt = torch.where(gt_segs <= config.dataset.num_seg_classes - config.dataset.num_classes, gt_segs, matched_gt)
        matched_gt = torch.where(gt_segs >= 255, gt_segs, matched_gt)
        if keep_inds is not None and pan_gt is not None:
            gt_masks = gt_masks[keep_inds]
            pan_gt = pan_gt[keep_inds]

        relation_mat = self.get_relation_gt(gt_masks)
        non_zero_num = 0
        new_index = None
        if sample:
            indicator = relation_mat.sum(dim=1)
            non_zero_num = (indicator > 0.5).sum()
            if non_zero_num > 1:
                new_index = (-indicator).argsort()
                relation_mat = relation_mat[new_index][:, new_index][:non_zero_num, :non_zero_num]
                gt_masks = gt_masks[new_index]
                pan_gt = pan_gt[new_index]

        for i in range(gt_masks.shape[0]):
            matched_gt[(pan_gt[[i], :, :] != 0) & (pan_gt[[i], :, :] != 255)] = i + self.num_seg_classes - self.num_inst_classes
            # matched_gt[(gt_masks[[i], :, :] != 0) & (gt_masks[[i], :, :] != 255)] = i + self.num_seg_classes - self.num_inst_classes
        if keep_inds is not None:
            matched_gt[matched_gt == -1] = self.num_seg_classes - self.num_inst_classes + gt_masks.shape[0]
        else:
            matched_gt[matched_gt == -1] = 255

        if not sample:
            return matched_gt, relation_mat
        else:
            return matched_gt, relation_mat, non_zero_num, new_index


class PanopticGTGenerate(nn.Module):

    def __init__(self, num_seg_classes, enable_void, class_mapping=None):
        super(PanopticGTGenerate, self).__init__()
        self.class_mapping = dict(zip(range(1, config.dataset.num_classes), range(num_seg_classes - config.dataset.num_classes + 1, num_seg_classes))) if class_mapping is None else class_mapping
        self.num_seg_classes = num_seg_classes
        self.num_inst_classes = len(self.class_mapping)
        self.enable_void = enable_void

    def forward(self, rois, bbox_pred, cls_score, label, gt_rois, cls_idx, seg_gt, mask_gt, im_shape):
        
        rois = rois.data.cpu().numpy()
        bbox_pred = bbox_pred.data.cpu().numpy()
        cls_score = cls_score.data.cpu().numpy()
        cls_pred = np.argmax(cls_score, axis=1)
        label = label.data.cpu().numpy()
        gt_rois = gt_rois.cpu().numpy()

        rois = rois[:, 1:]

        bbox_overlap = bbox_overlaps(rois, gt_rois[:, 1:])  # #rois x #gt_rois
        max_bbox_overlap = np.argmax(bbox_overlap, axis=1)
        max_overlap = np.ones((gt_rois.shape[0]), dtype=np.int32) * -1

        matched_gt = torch.ones_like(seg_gt) * -1
        matched_gt = torch.where(seg_gt <= config.dataset.num_seg_classes - config.dataset.num_classes, seg_gt, matched_gt)
        matched_gt = torch.where(seg_gt >= 255, seg_gt, matched_gt)

        keep = np.ones((rois.shape[0]), dtype=np.int32)

        for i in range(rois.shape[0]):
            if bbox_overlap[i, max_bbox_overlap[i]] > 0.5:
                if max_overlap[max_bbox_overlap[i]] == -1:
                    max_overlap[max_bbox_overlap[i]] = i
                elif bbox_overlap[max_overlap[max_bbox_overlap[i]], max_bbox_overlap[i]] > bbox_overlap[i, max_bbox_overlap[i]]:
                    keep[i] = 0
                else: 
                    keep[max_overlap[max_bbox_overlap[i]]] = 0
                    max_overlap[max_bbox_overlap[i]] = i
            elif cls_pred[i] == 0 and label[i] == 0:
                keep[i] = 0

        rois = rois[keep != 0]
        rois = np.hstack((np.zeros((rois.shape[0], 1)), rois))
        label = label[keep != 0]

        keep = np.cumsum(keep)
        if keep[-1] == 0:
            print(max_overlap)
            print(max_bbox_overlap)
            print(cls_pred)
            assert keep[-1] != 0

        for i in range(max_overlap.shape[0]):
            if max_overlap[i] != -1:
                roi = np.round(rois[keep[max_overlap[i]] - 1] / 4)
                mask_gt_i = mask_gt[[i]]
                matched_gt[mask_gt_i != 0] = int(keep[max_overlap[i]] - 1 + self.num_seg_classes - self.num_inst_classes)

        if config.train.panoptic_box_keep_fraction < 1:
            matched_gt[matched_gt == -1] = self.num_seg_classes - self.num_inst_classes + rois.shape[0]
        else:
            matched_gt[matched_gt == -1] = 255

        return torch.from_numpy(rois).to(matched_gt.device), torch.from_numpy(label).to(matched_gt.device), matched_gt



