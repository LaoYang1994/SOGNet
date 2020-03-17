from torch.nn.modules.module import Module
from ..functions.roialign import RoIAlignFunction


class RoIAlign(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIAlign, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features, rois)
