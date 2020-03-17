from torch.nn.modules.module import Module
from ..functions.proposal_target import ProposalTargetFunction

class ProposalTarget(Module):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction):
        super(ProposalTarget, self).__init__()
        self.proposal_target_function = ProposalTargetFunction(num_classes, batch_images, batch_rois, fg_fraction)

    def forward(self, rois, gt_boxes):
        return self.proposal_target_function(rois, gt_boxes)
