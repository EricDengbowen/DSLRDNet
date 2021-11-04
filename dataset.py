import os
from PIL import Image
import cv2
import torch
from torch.utils import data
import numpy as np
import random


class ImageDataTrain(data.Dataset):

    def __init__(self):
        self.sal_root = '/db/psxbd1/DUTS-TR/'
        self.sal_list_source = '/db/psxbd1/DUTS-TR/train_pair_edge.lst'

        with open(self.sal_list_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        sal_image = load_image(os.path.join(self.sal_root, self.sal_list[item].split()[0]))
        sal_label = load_sal_label(os.path.join(self.sal_root, self.sal_list[item].split()[1]))
        sal_edge = load_edge_label(os.path.join(self.sal_root, self.sal_list[item].split()[2]))
        sal_image, sal_label, sal_edge = cv_random_flip(sal_image, sal_label, sal_edge)
        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)
        sal_edge = torch.Tensor(sal_edge)

        sample = {'sal_image': sal_image, 'sal_label': sal_label, 'sal_edge': sal_edge}
        return sample

    def __len__(self):
        return self.sal_num

class ImageDataTest(data.Dataset):
    def __init__(self, test_mode='0', sal_mode='d'):
        if test_mode == 0:
            self.image_root = '/db/psxbd1/DUTS-TE/DUTS-TE-Image/'
            self.image_source = '/db/psxbd1/DUTS-TE/test.lst'
            self.test_fold = '/db/psxbd1/SaliencyMapResults/DUTS/'


        elif test_mode == 1:
            if sal_mode == 'e':
                self.image_root = '/db/psxbd1/ECSSD/images/'
                self.image_source = '/db/psxbd1/ECSSD/test.lst'
                self.test_fold = '/db/psxbd1/SaliencyMapResults/ECSSD/'
            elif sal_mode == 'p':
                self.image_root = '/db/psxbd1/PASCAL-S/images/'
                self.image_source = '/db/psxbd1/PASCAL-S/test.lst'
                self.test_fold = '/db/psxbd1/SaliencyMapResults/PASCAL-S/'
            elif sal_mode == 'd':
                self.image_root = '/db/psxbd1/DUTOMRON/images/'
                self.image_source = '/db/psxbd1/DUTOMRON/test.lst'
                self.test_fold = '/db/psxbd1/SaliencyMapResults/DUTOMRON/'
            elif sal_mode == 'h':
                self.image_root = '/db/psxbd1/HKU-IS/images/'
                self.image_source = '/db/psxbd1/HKU-IS/test.lst'
                self.test_fold = '/db/psxbd1/SaliencyMapResults/HKU-IS/'
            elif sal_mode == 's':
                self.image_root = '/db/psxbd1/SOD/images/'
                self.image_source = '/db/psxbd1/SOD/test.lst'
                self.test_fold = '/db/psxbd1/SaliencyMapResults/SOD/'
            elif sal_mode == 'm':
                self.image_root = '/db/psxbd1/MSOD/images/'
                self.image_source = '/db/psxbd1/MSOD/test.lst'
                self.test_fold = '/db/psxbd1/SaliencyMapResults/MSOD/'
            elif sal_mode == 'o':
                self.image_root = '/db/psxbd1/SOC/images/'
                self.image_source ='/db/psxbd1/SOC/test.lst'
                self.test_fold = '/db/psxbd1/SaliencyMapResults/SOC/'
            elif sal_mode == 't':
                self.image_root = '/db/psxbd1/DUTS-TE/DUTS-TE-Image/'
                self.image_source = '/db/psxbd1/DUTS-TE/test.lst'
                self.test_fold = '/db/psxbd1/SaliencyMapResults/DUTS/'

        with open(self.image_source, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.image_root, self.image_list[item].split()[0]))
        image = torch.Tensor(image)
        sample = {'image': image, 'name': self.image_list[item].split()[0], 'size': im_size}
        return sample

    def save_folder(self):
        return self.test_fold

    def __len__(self):
        return self.image_num


def load_image(pah):
    if not os.path.exists(pah):
        print('File Not Exists')
    im = cv2.imread(pah)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((123.68, 116.779, 103.939))
    in_ = in_.transpose((2, 0, 1))
    return in_


def load_sal_label(pah):

    if not os.path.exists(pah):
        print('File Not Exists')
    im = Image.open(pah)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    label = label / 255.
    label = label[np.newaxis, ...]
    return label


def load_edge_label(pah):

    if not os.path.exists(pah):
        print('File Not Exists')
    im = Image.open(pah)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    label = label / 255.
    label[np.where(label > 0.5)] = 1.
    label = label[np.newaxis, ...]
    return label


def cv_random_flip(img, label, edge):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:, :, ::-1].copy()
        label = label[:, :, ::-1].copy()
        edge = edge[:, :, ::-1].copy()
    return img, label, edge


def load_image_test(pah):
    if not os.path.exists(pah):
        print('File Not Exists')
    im = cv2.imread(pah)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((123.68, 116.779, 103.939))
    in_ = in_.transpose((2, 0, 1))
    return in_, im_size

def get_loader(batch_size, mode='train', num_thread=1, test_mode=0, sal_mode='e'):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain()
    else:
        dataset = ImageDataTest(test_mode=test_mode, sal_mode=sal_mode)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return data_loader, dataset

