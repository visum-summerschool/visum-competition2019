#import os
#import numpy as np
import torch
import torch.utils.data
#import h5py
#import pickle
#from PIL import Image
import torchvision
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from utils_.engine import train_one_epoch, evaluate
from utils_ import utils
from utils_ import transforms as T
#import visdom_utils
#from visdom import Visdom
from utils_.visum_utils import Dataset#, VisdomLinePlotter



DATA_DIR = '/data/DB/VISUM_newdata_baseline'
SAVE_MODEL = '/home/wjsilva19/VISUM_baseline/Models/fasterRCNN_model'


# Data augmentation
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

backbone = torchvision.models.mobilenet_v2(pretrained=True).features

backbone.out_channels = 1280

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)


# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=10,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

# See the model architecture
print(model)

# use our dataset and defined transformations
dataset = Dataset(DATA_DIR, 'RGB', 'train', transforms=get_transform(train=True))
dataset_val = Dataset(DATA_DIR, 'RGB', 'train', transforms=get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-100])
dataset_val = torch.utils.data.Subset(dataset_val, indices[-100:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=2, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=10,
                                               gamma=0.5)

num_epochs = 50

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    epoch_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluator = evaluate(model, data_loader_val, device=device)

torch.save(model, SAVE_MODEL)  




