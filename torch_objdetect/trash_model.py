import torch.nn as nn
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
class TrashModel(nn.Module):
    def __init__(self, num_classes):
        super(TrashModel, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     def __init__(self, num_classes):
#         super(TrashModel, self).__init__()
#         self.model = torch.load("/Users/i303138/Documents/Learning/MachineLearning/Projects/Lego/pytorch_objdetect/mb2-ssd-lite-mp-0_686.pth")
#         self.model.
    def forward(self, images, targets=None):
        return self.model.forward(images,targets)
    
#     def eval(self):
#         self.model.eval()
#     def state_dict(self):
#         return self.model.state_dict()
#     def load_state_dict(self,dicts):
#         self.model.load_state_dict(dicts)
       