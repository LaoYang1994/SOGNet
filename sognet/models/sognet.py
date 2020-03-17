import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sognet.config.config import config
from sognet.models.resnet import get_params, resnet_rcnn, ResNetBackbone
from sognet.models.fpn import FPN
from sognet.models.rpn import RPN, RPNLoss
from sognet.models.rcnn import RCNN, MaskBranch, RCNNLoss, MaskRCNNLoss
from sognet.models.fcn import FCNHead
from sognet.models.relation import RelationHead, RelationLoss
from sognet.operators.modules.pyramid_proposal import PyramidProposal
from sognet.operators.modules.proposal_mask_target import ProposalMaskTarget
from sognet.operators.modules.mask_roi import MaskROI
from sognet.operators.modules.unary_logits import MaskTerm, SegTerm
from sognet.operators.modules.mask_removal import MaskRemoval
from sognet.operators.modules.mask_matching import MaskMatching
if config.train.use_horovod and config.network.use_syncbn:
    from sognet.operators.modules.distbatchnorm import BatchNorm2d



class SOGNet(resnet_rcnn):
    
    def __init__(self, backbone_depth):
        super(SOGNet, self).__init__()

        self.num_classes = config.dataset.num_classes
        self.num_seg_classes = config.dataset.num_seg_classes
        self.num_reg_classes = (2 if config.network.cls_agnostic_bbox_reg else config.dataset.num_classes)

        # backbone net
        self.resnet_backbone = ResNetBackbone(backbone_depth)
        # FPN, RPN, Instance Head and Semantic Head
        self.fpn = FPN(feature_dim=config.network.fpn_feature_dim, with_norm=config.network.fpn_with_norm,
                        upsample_method=config.network.fpn_upsample_method)
        self.rpn =  RPN(num_anchors=config.network.num_anchors, input_dim=config.network.fpn_feature_dim)
        self.rcnn = RCNN(self.num_classes, self.num_reg_classes, dim_in=config.network.fpn_feature_dim, 
                         with_norm=config.network.rcnn_with_norm)
        self.mask_branch = MaskBranch(self.num_classes, dim_in=config.network.fpn_feature_dim,
                                      with_norm=config.network.rcnn_with_norm)
        self.fcn_head = eval(config.network.fcn_head)(config.network.fpn_feature_dim, self.num_seg_classes, 
                                                      num_layers=config.network.fcn_num_layers,
                                                      with_norm=config.network.fcn_with_norm, upsample_rate=4,
                                                      with_roi_loss=config.train.fcn_with_roi_loss)
        self.mask_roi = MaskROI(clip_boxes=True, bbox_class_agnostic=False, top_n=config.test.max_det, 
                                num_classes=self.num_classes, score_thresh=config.test.score_thresh)

        # Panoptic Head
        # relation head
        self.relation_head = RelationHead()
        # param for training
        self.box_keep_fraction = config.train.panoptic_box_keep_fraction
        self.enable_void = config.train.panoptic_box_keep_fraction < 1

        self.mask_roi_panoptic = MaskROI(clip_boxes=True, bbox_class_agnostic=False, top_n=config.test.max_det, 
                                         num_classes=self.num_classes, nms_thresh=0.5, class_agnostic=True, score_thresh=config.test.panoptic_score_thresh)
        self.mask_removal = MaskRemoval(fraction_threshold=0.3)
        self.seg_term = SegTerm(config.dataset.num_seg_classes)
        self.mask_term = MaskTerm(config.dataset.num_seg_classes, box_scale=1/4.0)
        self.mask_matching = MaskMatching(config.dataset.num_seg_classes, enable_void=self.enable_void)

        # # Loss layer
        self.rpn_loss = RPNLoss(config.train.rpn_batch_size * config.train.batch_size)
        self.mask_rcnn_loss = MaskRCNNLoss(config.train.batch_rois * config.train.batch_size)
        self.fcn_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.panoptic_loss = nn.CrossEntropyLoss(ignore_index=255, reduce=False)
        self.relation_loss = RelationLoss()
        if config.train.fcn_with_roi_loss:
            self.fcn_roi_loss = nn.CrossEntropyLoss(ignore_index=255, reduce=False)
        self.initialize()
        self.fcn_cls_mapping = dict(zip(range(1, config.dataset.num_classes), range(config.dataset.num_seg_classes - config.dataset.num_classes + 1, config.dataset.num_seg_classes)))

    def initialize(self):
        pass

    def forward(self, data, label=None):

        res2, res3, res4, res5 = self.resnet_backbone(data['data'])
        fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6 = self.fpn(res2, res3, res4, res5)

        rpn_cls_score, rpn_cls_prob, rpn_bbox_pred = [], [], []
        for feat in [fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6]:
            rpn_cls_score_p, rpn_bbox_pred_p, rpn_cls_prob_p = self.rpn(feat)
            rpn_cls_score.append(rpn_cls_score_p)
            rpn_cls_prob.append(rpn_cls_prob_p)
            rpn_bbox_pred.append(rpn_bbox_pred_p)

        if label is not None:
            self.pyramid_proposal = PyramidProposal(feat_stride=config.network.rpn_feat_stride, scales=config.network.anchor_scales,
                                                    ratios=config.network.anchor_ratios, rpn_pre_nms_top_n=config.train.rpn_pre_nms_top_n,
                                                    rpn_post_nms_top_n=config.train.rpn_post_nms_top_n, threshold=config.train.rpn_nms_thresh,
                                                    rpn_min_size=config.train.rpn_min_size, individual_proposals=config.train.rpn_individual_proposals)
            self.proposal_target = ProposalMaskTarget(num_classes=self.num_reg_classes,
                                                      batch_images=config.train.batch_size, batch_rois=config.train.batch_rois,
                                                      fg_fraction=config.train.fg_fraction, mask_size=config.network.mask_size,
                                                      binary_thresh=config.network.binary_thresh)
            rois, _ = self.pyramid_proposal(rpn_cls_prob, rpn_bbox_pred, data['im_info'])
            rois, cls_label, bbox_target, \
                bbox_inside_weight, bbox_outside_weight, mask_rois, \
                    mask_target, roi_has_mask, nongt_inds \
                        = self.proposal_target(rois, label['roidb'], data['im_info'])
        else:
            self.pyramid_proposal = PyramidProposal(feat_stride=config.network.rpn_feat_stride, scales=config.network.anchor_scales,
                                                    ratios=config.network.anchor_ratios, rpn_pre_nms_top_n=config.test.rpn_pre_nms_top_n,
                                                    rpn_post_nms_top_n=config.test.rpn_post_nms_top_n, threshold=config.test.rpn_nms_thresh,
                                                    rpn_min_size=config.test.rpn_min_size, individual_proposals=config.train.rpn_individual_proposals)
            rois, _ = self.pyramid_proposal(rpn_cls_prob, rpn_bbox_pred, data['im_info'])

        if label is not None and config.train.fcn_with_roi_loss:
            fcn_rois, _ = self.get_gt_rois(label['roidb'], data['im_info'])
            fcn_rois = fcn_rois.to(rois.device)
            fcn_output = self.fcn_head(*[fpn_p2, fpn_p3, fpn_p4, fpn_p5, fcn_rois])
        else:
            fcn_output = self.fcn_head(*[fpn_p2, fpn_p3, fpn_p4, fpn_p5])

        if label is not None:

            # RPN loss
            rpn_cls_loss, rpn_bbox_loss = self.rpn_loss(rpn_cls_score, rpn_bbox_pred, label)

            # Semantic head loss
            fcn_loss = self.fcn_loss(fcn_output['fcn_output'], label['seg_gt'])
            if config.train.fcn_with_roi_loss:
                fcn_roi_loss = self.fcn_roi_loss(fcn_output['fcn_roi_score'], label['seg_roi_gt'])
                fcn_roi_loss = fcn_roi_loss.mean()

            # Instance head loss
            rcnn_output = self.rcnn([fpn_p2, fpn_p3, fpn_p4, fpn_p5], rois)
            cls_score, bbox_pred = rcnn_output['cls_score'], rcnn_output['bbox_pred']
            mask_score = self.mask_branch([fpn_p2, fpn_p3, fpn_p4, fpn_p5], mask_rois)
            cls_loss, bbox_loss, mask_loss, rcnn_acc = \
                self.mask_rcnn_loss(cls_score, bbox_pred, mask_score,
                                    cls_label, bbox_target, bbox_inside_weight, bbox_outside_weight, mask_target)

            # Panoptic head

            # extract gt rois for panoptic head
            gt_rois, cls_idx = self.get_gt_rois(label['roidb'], data['im_info'])
            if self.enable_void:
                keep_inds = np.random.choice(gt_rois.shape[0], max(int(gt_rois.shape[0] * self.box_keep_fraction), 1), replace=False)
                gt_rois = gt_rois[keep_inds]
                cls_idx = cls_idx[keep_inds]
            gt_rois, cls_idx = gt_rois.to(rois.device), cls_idx.to(rois.device)

            # generate gt for panoptic head


            if self.num_classes == 81:
                pan_gt = label['matched_mask']
            elif self.num_classes == 9:
                pan_gt = label['mask_gt']

            with torch.no_grad():
                if not config.relation.sample:
                    if self.enable_void:
                        panoptic_gt, relation_gt = self.mask_matching(label['seg_gt_4x'], label['mask_gt'], keep_inds=keep_inds, pan_gt=pan_gt)
                    else:
                        panoptic_gt, relation_gt = self.mask_matching(label['seg_gt_4x'], label['mask_gt'], pan_gt=pan_gt)
                else:
                    if self.enable_void:
                        panoptic_gt, relation_gt, non_zero_num, new_index \
                            = self.mask_matching(label['seg_gt_4x'], label['mask_gt'], keep_inds=keep_inds, sample=True, pan_gt=pan_gt)
                    else:
                        panoptic_gt, relation_gt, non_zero_num, new_index \
                            = self.mask_matching(label['seg_gt_4x'], label['mask_gt'], sample=True, pan_gt=pan_gt)


            # Calc mask logits with gt rois
            mask_score = self.mask_branch([fpn_p2, fpn_p3, fpn_p4, fpn_p5], gt_rois)
            mask_score = mask_score.gather(1, cls_idx.view(-1, 1, 1, 1).expand(-1, -1, config.network.mask_size, config.network.mask_size))
            mask_logits = self.mask_term(mask_score, gt_rois, cls_idx, fcn_output['fcn_score'])
            # non_zero_num = min(80, non_zero_num)

            relation_mat = None
            if config.relation.sample:
                # if sample, then resort the mask
                if non_zero_num > 0:
                    cls_idx = cls_idx[new_index]
                    gt_rois = gt_rois[new_index]
                    mask_logits = mask_logits[:, new_index, ...]
                    relation_gt = relation_gt[:non_zero_num, :non_zero_num]
                    # relation = relation[new_index][:, new_index][:non_zero_num, :non_zero_num]
                    relation_part = mask_logits[:, :non_zero_num, ...]
                    no_relation_part = mask_logits[:, non_zero_num:, ...]

                    relation_part, relation_mat = self.relation_head(relation_part, \
                        copy.deepcopy(gt_rois[:non_zero_num]), copy.deepcopy(cls_idx[:non_zero_num]))
                    mask_logits = torch.cat([relation_part, no_relation_part], dim=1)
            else:
                if relation_gt.sum() > 1:
                    mask_logits, relation_mat = self.relation_head(mask_logits, \
                        copy.deepcopy(gt_rois), copy.deepcopy(cls_idx))

            # Calc panoptic logits
            seg_logits, seg_inst_logits = self.seg_term(cls_idx, fcn_output['fcn_score'], gt_rois)

            if self.enable_void:
                void_logits = torch.max(fcn_output['fcn_score'][:, (config.dataset.num_seg_classes - config.dataset.num_classes + 1):, ...], dim=1, keepdim=True)[0] - torch.max(seg_inst_logits, dim=1, keepdim=True)[0]
                if config.train.x2mask:
                    inst_logits = mask_logits + 2 * torch.mul(seg_inst_logits, torch.sigmoid(mask_logits))
                else:
                    inst_logits = seg_inst_logits + mask_logits
                panoptic_logits = torch.cat([seg_logits, inst_logits, void_logits], dim=1)
            else:
                if config.train.x2mask:
                    panoptic_logits = torch.cat([seg_logits, mask_logits + 2 * torch.mul(seg_inst_logits, torch.sigmoid(mask_logits))], dim=1)
                else:
                    panoptic_logits = torch.cat([seg_logits, (seg_inst_logits + mask_logits)], dim=1)

            if relation_mat is not None:
                relation_loss = self.relation_loss(relation_mat, relation_gt)
            else:
                relation_loss = torch.tensor(0).float().to(seg_logits.device)

            # Panoptic head loss
            panoptic_acc = self.calc_panoptic_acc(panoptic_logits, panoptic_gt)
            panoptic_loss = self.panoptic_loss(panoptic_logits, panoptic_gt)
            panoptic_loss = panoptic_loss.mean()

            output = {
                'rpn_cls_loss': rpn_cls_loss.unsqueeze(0),
                'rpn_bbox_loss': rpn_bbox_loss.unsqueeze(0),
                'cls_loss': cls_loss.unsqueeze(0),
                'bbox_loss': bbox_loss.unsqueeze(0),
                'mask_loss': mask_loss.unsqueeze(0),
                'fcn_loss': fcn_loss.unsqueeze(0),
                'panoptic_loss': panoptic_loss.unsqueeze(0),
                'relation_loss': relation_loss.unsqueeze(0),
                'rcnn_accuracy': rcnn_acc.unsqueeze(0),
                'panoptic_accuracy': panoptic_acc.unsqueeze(0),
            }
            if config.train.fcn_with_roi_loss:
                output.update({'fcn_roi_loss': fcn_roi_loss})

            return output

        else:

            rcnn_output = self.rcnn([fpn_p2, fpn_p3, fpn_p4, fpn_p5], rois)
            cls_score, bbox_pred = rcnn_output['cls_score'], rcnn_output['bbox_pred']
            cls_prob = F.softmax(cls_score, dim=1)

            cls_prob_all, mask_rois, cls_idx = self.mask_roi(rois, bbox_pred, cls_prob, data['im_info'])
            mask_score = self.mask_branch([fpn_p2, fpn_p3, fpn_p4, fpn_p5], mask_rois)
            mask_prob = torch.sigmoid(mask_score)

            # get mask rcnn output (optional)
            results = {
                'cls_probs': cls_prob_all,
                'pred_boxes': mask_rois,
                'mask_probs': mask_prob,
                'fcn_outputs': torch.max(fcn_output['fcn_output'], dim=1)[1],
                'cls_inds': cls_idx
            }

            # get mask_logits
            cls_prob, mask_rois, cls_idx = self.mask_roi_panoptic(rois, bbox_pred, cls_prob, data['im_info'])
            mask_score = self.mask_branch([fpn_p2, fpn_p3, fpn_p4, fpn_p5], mask_rois)
            mask_score = mask_score.gather(1, cls_idx.view(-1, 1, 1, 1).expand(-1, -1, config.network.mask_size, config.network.mask_size))

            # get panoptic logits
            # keep_inds, mask_logits = self.mask_removal(mask_rois[:, 1:]/4, cls_prob, mask_score, cls_idx, fcn_output['fcn_score'].shape[2:])     size /4
            keep_inds, mask_logits = self.mask_removal(mask_rois[:, 1:], cls_prob, mask_score, cls_idx, fcn_output['fcn_output'].shape[2:])      # original size
            # keep_inds, mask_logits = self.mask_removal(mask_rois[:, 1:] / 2, cls_prob, mask_score, cls_idx, fcn_output['fcn_output'].shape[2:])  size /2

            mask_rois = mask_rois[keep_inds]
            cls_idx = cls_idx[keep_inds]
            cls_prob = cls_prob[keep_inds]

            if config.relation.sample:
                new_index, non_zero_num = self.get_new_index(mask_logits, data['im_info'])

                mask_logits = mask_logits[:, new_index]
                cls_idx = cls_idx[new_index]
                mask_rois = mask_rois[new_index]
                cls_prob = cls_prob[new_index]
                if non_zero_num > 0:
                    relation_part = mask_logits[:, :non_zero_num]
                    no_relation_part = mask_logits[:, non_zero_num:]

                    ####   new inf   ####
                    mask_rois_relation = mask_rois[:non_zero_num]
                    cls_idx_relation = cls_idx[:non_zero_num]

                    for ins_id1 in range(non_zero_num-1):
                        for ins_id2 in range(ins_id1+1, non_zero_num):
                            selected = torch.cat([torch.unsqueeze(relation_part[:, ins_id1], dim=1), torch.unsqueeze(relation_part[:, ins_id2], dim=1)], dim=1)
                            mask_rois_selected = torch.cat([torch.unsqueeze(mask_rois_relation[ins_id1,:], dim=0), torch.unsqueeze(mask_rois_relation[ins_id2,:], dim=0)], dim=0)
                            cls_idx_selected = torch.cat([torch.unsqueeze(cls_idx_relation[ins_id1], dim=0), torch.unsqueeze(cls_idx_relation[ins_id2], dim=0)], dim=0)
                            
                            selected, _ = self.relation_head(selected, mask_rois_selected, cls_idx_selected)
                            relation_part[:, ins_id1] = selected[:, 0]
                            relation_part[:, ins_id2] = selected[:, 1]

                    # relation_part, _ = self.relation_head(relation_part, mask_rois[:non_zero_num], cls_idx[:non_zero_num])
                    mask_logits = torch.cat([relation_part, no_relation_part], dim=1)
            else:
                num_ins = mask_logits.size(1)
                new_index = [x for x in range(num_ins)]
                np.random.shuffle(new_index)
                new_index = torch.tensor(new_index).to(mask_logits.device)
                mask_logits = mask_logits[:, new_index]
                mask_rois = mask_rois[new_index]
                cls_idx = cls_idx[new_index]
                cls_prob = cls_prob[new_index]

                if mask_logits.size(1) > 4:
                    non_zero_num = int(mask_logits.size(1) / 2)
                    relation_part = mask_logits[:, :non_zero_num]
                    no_relation_part = mask_logits[:, non_zero_num:]
                    relation_part, _ = self.relation_head(relation_part, mask_rois[:non_zero_num], cls_idx[:non_zero_num])
                    mask_logits = torch.cat([relation_part, no_relation_part], dim=1)
                else:
                    mask_logits, _ = self.relation_head(mask_logits, mask_rois, cls_idx)

            # seg_logits, seg_inst_logits = self.seg_term(cls_idx, fcn_output['fcn_score'], mask_rois)        for size / 4
            # seg_logits, seg_inst_logits = self.seg_term(cls_idx, fcn_output['fcn_output'], mask_rois * 2.0)   for size /2
            seg_logits, seg_inst_logits = self.seg_term(cls_idx, fcn_output['fcn_output'], mask_rois * 4.0)   # for orginal size
            results.update({
                'panoptic_cls_inds': cls_idx, 
                'panoptic_cls_probs': cls_prob
            })

            if self.enable_void:
                # void_logits = torch.max(fcn_output['fcn_score'][:, (config.dataset.num_seg_classes - config.dataset.num_classes + 1):, ...], dim=1, keepdim=True)[0] - torch.max(seg_inst_logits, dim=1, keepdim=True)[0]
                void_logits = torch.max(fcn_output['fcn_output'][:, (config.dataset.num_seg_classes - config.dataset.num_classes + 1):, ...], dim=1, keepdim=True)[0] - torch.max(seg_inst_logits, dim=1, keepdim=True)[0]
                if config.test.x2mask:
                    inst_logits = mask_logits + 2 * torch.mul(seg_inst_logits, torch.sigmoid(mask_logits))#(seg_inst_logits + mask_logits)
                else:
                    inst_logits = (seg_inst_logits + mask_logits)
                panoptic_logits = torch.cat([seg_logits, inst_logits, void_logits], dim=1)
                void_id = panoptic_logits.shape[1] - 1
                # panoptic_logits = F.interpolate(panoptic_logits, None, 2, mode='bilinear', align_corners=False)
                panoptic_output = torch.max(panoptic_logits, dim=1)[1]
                panoptic_output[panoptic_output == void_id] = 255
            else:
                if config.test.x2mask:
                    inst_logits = mask_logits + 2 * torch.mul(seg_inst_logits, torch.sigmoid(mask_logits))#(seg_inst_logits + mask_logits)
                else:
                    inst_logits = (seg_inst_logits + mask_logits)
                panoptic_logits = torch.cat([seg_logits, (inst_logits)], dim=1)
                # panoptic_logits = F.interpolate(panoptic_logits, None, 2, mode='bilinear', align_corners=False)
                panoptic_output = torch.max(F.softmax(panoptic_logits, dim=1), dim=1)[1]

            results.update({
                'panoptic_outputs': panoptic_output,
            })
            return results

    def calc_panoptic_acc(self, panoptic_logits, gt):
        _, output_cls = torch.max(panoptic_logits.data, 1, keepdim=True)
        ignore = (gt == 255).long().sum()
        correct = (output_cls.view(-1) == gt.data.view(-1)).long().sum()
        total = (gt.view(-1).shape[0]) - ignore
        assert total != 0
        panoptic_acc = correct.float() / total.float()
        return panoptic_acc


    def get_params_lr(self):
        ret = []
        gn_params = []
        gn_params_name = []
        for n, m in self.named_modules():
            if isinstance(m, nn.GroupNorm) or (config.train.use_horovod and config.network.use_syncbn and isinstance(m, BatchNorm2d)):
                gn_params.append(m.weight)
                gn_params.append(m.bias)
                gn_params_name.append(n + '.weight')
                gn_params_name.append(n + '.bias')

        ret.append({'params': gn_params, 'lr': 1, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['resnet_backbone.res3', 'resnet_backbone.res4', 'resnet_backbone.res5'], ['weight'])], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['resnet_backbone.res3', 'resnet_backbone.res4', 'resnet_backbone.res5'], ['bias'])], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['fpn'], ['weight'], exclude=gn_params_name)], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['fpn'], ['bias'], exclude=gn_params_name)], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['rcnn'], ['weight'], exclude=gn_params_name)], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['rcnn'], ['bias'], exclude=gn_params_name)], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['mask_branch'], ['weight'], exclude=gn_params_name)], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['mask_branch'], ['bias'], exclude=gn_params_name)], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['rpn'], ['weight'])], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['rpn'], ['bias'])], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['fcn_head'], ['weight'], exclude=gn_params_name)], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['fcn_head'], ['bias'], exclude=gn_params_name)], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['relation_head'], ['weight'], exclude=gn_params_name)], 'lr': 1})

        return ret

    def get_new_index(self, logits, im_info):
        h, w, _ = im_info[0]
        h = int(h)
        w = int(w)
        logits = logits[0]
        assert logits.size(1) >= h
        assert logits.size(2) >= w
        binary_mask = torch.zeros((logits.size(1), logits.size(2))).float().cuda()
        binary_mask[:h, :w] = 1.0
        logits = logits * binary_mask

        n = logits.size(0)
        mask = (logits > 0).float()
        area = mask.view(n, -1).sum(dim=1)
        inter = torch.zeros(n, n).type_as(logits)
        for i in range(n):
            t_mask = mask[i]
            t_inter = mask * t_mask
            inter[i] = t_inter.view(n, -1).sum(-1)
        # inter = mask[:, None, ...] * mask
        # inter = inter.view(n, n, -1).sum(dim=2)
        min_area = torch.min(area[:, None], area)
        iou = (inter / min_area) - torch.eye(n).float().cuda()
        relation_mat = (iou > 0.1).float()
        indicator = relation_mat.sum(dim=1)
        new_index = (-indicator).argsort()
        non_zero_num = (indicator > 0).sum()
        return new_index, non_zero_num

    def get_gt_rois(self, roidb, im_info):
        gt_inds = np.where((roidb['gt_classes'] > 0) & (roidb['is_crowd'] == 0))[0]
        rois = roidb['boxes'][gt_inds] * im_info[0, 2]
        cls_idx = roidb['gt_classes'][gt_inds]
        return torch.from_numpy(np.hstack((np.zeros((rois.shape[0], 1), dtype=np.float32), rois))), torch.from_numpy(cls_idx).long()

def resnet_101_sognet():
    return SOGNet([3, 4, 23, 3])

def resnet_50_sognet():
    return SOGNet([3, 4, 6, 3])
