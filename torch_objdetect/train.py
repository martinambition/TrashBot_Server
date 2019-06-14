import torch
import os
import xml.etree.ElementTree as ET
import numpy as np
import collections

from trash_lite_model import TrashLiteModel
from engine import train_one_epoch, evaluate
from PIL import Image
import utils
import transforms as T


class TrashDataset(object):
    def __init__(self, root, transforms, obj_types):
        self.root = root
        self.obj_types = obj_types
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        annotation_files = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
        self.annotation = []
        for afile in annotation_files:
            file_path = os.path.join(root, "Annotations", afile)
            anno_dict = self.parse_voc_xml(
                ET.parse(file_path).getroot())
            self.annotation.append(anno_dict)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def __getitem__(self, idx):
        annotation = self.annotation[idx]['annotation']
        img_path = os.path.join(self.root, "JPEGImages", annotation['folder'], annotation['filename'])
        img = Image.open(img_path).convert("RGB")
        num_objs = len(annotation['object'])
        boxes = []
        labels = []
        if not isinstance(annotation['object'], list):
            annotation['object'] = [annotation['object']]
        for obj in annotation['object']:
            bnbox = obj['bndbox']
            boxes.append([float(bnbox['xmin']), float(bnbox['ymin']), float(bnbox['xmax']), float(bnbox['ymax'])])
            labels.append(self.obj_types[obj['name']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.LongTensor(labels)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.annotation)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

num_classes = 3 #Background, Shoe, Cola
classes = {"Shoes":1,"Cola":2}
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = TrashLiteModel(num_classes)
dataset = TrashDataset("Dataset",get_transform(train=True),classes)
dataset_test = TrashDataset('Dataset', get_transform(train=False),classes)

# # split the dataset in train and test set
# indices = torch.randperm(len(dataset)).tolist()
# dataset = torch.utils.data.Subset(dataset, indices[:-50])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

model.to(device)
num_epochs = 10
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    print("Epoch: "+ str(epoch))
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    #evaluate on the test dataset
    #evaluate(model, data_loader_test, device=device)
model.eval()
torch.save(model.state_dict(), "./trash.pt")