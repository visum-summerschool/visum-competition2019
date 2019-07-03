import os
import numpy as np
import torch
import torch.utils.data
import h5py
import pickle
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from nms import nms
from visum_utils import VisumDataset


REJECT_THRESHOLD = 0.5
NMS_THRESHOLD = 0.1
SAVED_MODEL = '/home/master/VISUM_baseline/fasterRCNN_model'
OUTPUT_FILE = '/home/master/VISUM_baseline/predictions.csv'
DATA_DIR = '/home/master/dataset/train'


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Load datasets
test_data = VisumDataset(DATA_DIR, 'rgb', transforms=get_transform(False))
#test_data = Dataset(DATA_DIR, 'RGB', 'final_test', transforms=get_transform(False))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torch.load(SAVED_MODEL)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

predictions = list()
for i, img, _ in enumerate(test_loader):
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    boxes = np.array(prediction[0]['boxes'].cpu())
    labels = list(prediction[0]['labels'].cpu())
    scores = list(prediction[0]['scores'].cpu())

    nms_boxes, nms_labels = nms(boxes, labels, NMS_THRESHOLD)

    for bb in range(len(nms_labels)):
        pred = [ii]  # image number
        pred = np.concatenate((pred, list(nms_boxes[bb, :])))  # bounding box
        if scores[bb] >= REJECT_THRESHOLD:
            pred = np.concatenate((pred, [nms_labels[bb]]))  # object label
        else:
            pred = np.concatenate((pred, [-1]))  # Rejects to classify
        #pred = np.concatenate((pred, [scores[bb]]))  # BEST CLASS SCORE
        predictions.append(pred)
        print(ii)

np.savetxt(OUTPUT_FILE, predictions, delimiter=",")

