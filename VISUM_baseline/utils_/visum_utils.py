import os
import h5py
import numpy as np
import torch
from PIL import Image
import utils_.visdom_utils
from visdom import Visdom


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, modality, dset, transforms=None):
        self.root = root
        self.transforms = transforms
        self.modality = modality
        # load all image files, sorting them to
        # ensure that they are aligned
        self.img_file = os.path.join(root, 'bosch_' + dset + '.hdf5')
        self.ann_file = os.path.join(root, 'bosch_' + dset + '_' + modality + '_ann.csv')
        with h5py.File(self.img_file, 'r') as hf:
            self.n_imgs = hf[modality].shape[0]
        with open(self.ann_file, 'rb') as hf:
            self.ann = np.loadtxt(self.ann_file, delimiter=',')
            
    def __getitem__(self, idx):
        # load images and masks
        with h5py.File(self.img_file, 'r') as hf:
            img = hf[self.modality][idx]
        # get bounding box coordinates for each mask
        img = Image.fromarray((img*255).astype('uint8'))

        ann = self.ann[self.ann[:,0] == idx]
        num_objs = len(ann)
        boxes = list()
        labels = list()
        for ii in range(num_objs):
            boxes.append(ann[ii][1:5])
            labels.append(ann[ii][5])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        num_objs = torch.tensor([num_objs])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])

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
        return len(np.unique(self.ann[:, 0]))


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')