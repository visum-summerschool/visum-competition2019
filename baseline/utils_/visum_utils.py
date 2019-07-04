import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from visdom import Visdom
import csv


class VisumData(Dataset):
    def __init__(self, path, modality='all', mode='train', transforms=None):
        self.path = path
        self.transforms = transforms
        self.mode = mode

        assert modality in ['rgb', 'nir', 'all'], \
            'modality should be on of the following: \'rgb\', \'nir\', \'all\''
        self.modality = modality

        if self.modality in ['rgb', 'nir']:  # load only RGB or NIR images
            self.image_files = [f for f in os.listdir(path) if ('.jpg' in f) and (self.modality.upper() in f)]
        else:  # load all images (RGB and NIR)
            self.image_files = [f for f in os.listdir(path) if '.jpg' in f]

        if self.mode == 'train':
            self.annotations = dict()
            with open(os.path.join(self.path, 'annotation.csv')) as csv_file:
                for row in csv.reader(csv_file, delimiter=','):
                    file_name = row[0]
                    obj = [float(value) for value in row[1:5]]
                    obj.append(int(row[5]))

                    if file_name in self.annotations:
                        self.annotations[file_name].append(obj)
                    else:
                        self.annotations[file_name] = [obj]

            self.class_names = {
                0: 'book', 1: 'bottle', 2: 'box', 3: 'cellphone',
                4: 'cosmetics', 5: 'glasses', 6: 'headphones', 7: 'keys',
                8: 'wallet', 9: 'watch', -1: 'n.a.'}

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        img = Image.open(os.path.join(self.path, file_name))

        if self.mode == 'train':
            ann_key = self.image_files[idx].replace('NIR', 'RGB')
            ann = self.annotations.get(ann_key, [])

            num_objs = len(ann)
            boxes = list()
            labels = list()
            for ii in range(num_objs):
                boxes.append(ann[ii][0:4])
                labels.append(ann[ii][4])

            if num_objs > 0:
                image_id = torch.tensor([idx])
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

                target = {}
                target["image_id"] = image_id
                target["boxes"] = boxes
                target["labels"] = labels
                target["area"] = area
                target["iscrowd"] = iscrowd
            else:
                target = None
        else:
            target = None

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, file_name

    def __len__(self):
        return len(self.image_files)


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