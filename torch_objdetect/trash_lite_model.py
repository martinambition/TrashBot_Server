import torch.nn as nn
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator



class TrashLiteModel(nn.Module):
    def __init__(self, num_classes):
        super(TrashLiteModel, self).__init__()
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                        output_size=7,
                                                        sampling_ratio=2)

        # put the pieces together inside a FasterRCNN model
        self.model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)

    def forward(self, images, targets=None):
        return self.model.forward(images, targets)

# def eval(self):
#         self.model.eval()
#     def state_dict(self):
#         return self.model.state_dict()
#     def load_state_dict(self,dicts):
#         self.model.load_state_dict(dicts)
