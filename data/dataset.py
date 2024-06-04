from __future__ import print_function

import numpy as np
from skimage import color
from PIL import Image

import torch
import torchvision.datasets as datasets
import os
import random
import json
from torch.utils.data import Dataset

from torchvision import transforms

class ImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, target, index


class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img


class RGB2HSV(object):
    """Convert RGB PIL image to ndarray HSV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hsv(img)
        return img


class RGB2HED(object):
    """Convert RGB PIL image to ndarray HED."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hed(img)
        return img


class RGB2LUV(object):
    """Convert RGB PIL image to ndarray LUV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2luv(img)
        return img


class RGB2YUV(object):
    """Convert RGB PIL image to ndarray YUV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yuv(img)
        return img


class RGB2XYZ(object):
    """Convert RGB PIL image to ndarray XYZ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2xyz(img)
        return img


class RGB2YCbCr(object):
    """Convert RGB PIL image to ndarray YCbCr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ycbcr(img)
        return img


class RGB2YDbDr(object):
    """Convert RGB PIL image to ndarray YDbDr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ydbdr(img)
        return img


class RGB2YPbPr(object):
    """Convert RGB PIL image to ndarray YPbPr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ypbpr(img)
        return img


class RGB2YIQ(object):
    """Convert RGB PIL image to ndarray YIQ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yiq(img)
        return img


class RGB2CIERGB(object):
    """Convert RGB PIL image to ndarray RGBCIE."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2rgbcie(img)
        return img


class TouchFolderLabel(Dataset):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False, mode='train', label='full', data_amount=100):
        self.two_crop = two_crop
        self.dataroot = '/home/chenning/Datasets/TAG'
        self.mode = mode
        if mode == 'train':
            with open(os.path.join(root, 'train.txt'),'r') as f:
                data = f.read().split('\n')
        elif mode == 'test':
            with open(os.path.join(root, 'test.txt'),'r') as f:
                data = f.read().split('\n')
        elif mode == 'pretrain':
            with open(os.path.join(root, 'pretrain.txt'),'r') as f:
                data = f.read().split('\n')
        else:
            print('Mode other than train and test')
            exit()
        
        if mode == 'train' and label == 'rough':
            with open(os.path.join(root, 'train_rough.txt'),'r') as f:
                data = f.read().split('\n')
        
        if mode == 'test' and label == 'rough':
            with open(os.path.join(root, 'test_rough.txt'),'r') as f:
                data = f.read().split('\n')

        
        self.length = len(data)
        self.env = data
        self.transform = transform
        self.target_transform = target_transform
        self.label = label



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        
        assert index < self.length,'index_A range error'

        raw, target = self.env[index].strip().split(',') # mother path for A
        target = int(target)
        if self.label == 'hard':
            if target == 7 or target == 8 or target == 9 or target == 11 or target == 13:
                target = 1
            else:
                target = 0
        
        #idx = os.path.basename(raw)
        # dir = self.dataroot + raw[:16]

        # # load image and gelsight
        # A_img_path = os.path.join(dir, 'video_frame', idx)
        # A_gelsight_path = os.path.join(dir, 'gelsight_frame', idx)
        idx = raw.replace('/', '__')
        #A_img_path = os.path.join(self.dataroot, 'vision/train_ori', idx)
        A_gelsight_path = os.path.join(self.dataroot, 'touch/train_ori', idx)
        
        #A_img = Image.open(A_img_path).convert('RGB')
        A_gel = Image.open(A_gelsight_path).convert('RGB')
        

        if self.transform is not None:
            #A_img = self.transform(A_img)
            A_gel = self.transform(A_gel)

        #out = torch.cat((A_img, A_gel), dim=0)
        
        if self.mode == 'pretrain':
            return A_gel, target, index

        return A_gel, target
    
    def __len__(self):
        """Return the total number of images."""
        return self.length


class FeelingLabel(Dataset):
    """datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False, mode='train', label='full', data_amount=100):
        self.two_crop = two_crop
        self.dataroot = '/home/chenning/Datasets/feeling/data'
        self.mode = mode
        if mode == 'train':
            with open(os.path.join(root, 'train_info.json'),'r',encoding='utf-8') as f:
                data = f.readlines()
            # with open(os.path.join(root, 'valid_info.json'),'r',encoding='utf-8') as f:
            #     valid_data = f.readlines()
        elif mode == 'test':
            with open(os.path.join(root, 'test_info.json'),'r') as f:
                data = f.readlines()
        else:
            print('Mode other than train and test')
            exit()

        
        self.length = len(data)
        self.env = data
        self.transform = transform
        self.target_transform = target_transform
        self.label = label



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        
        assert index < self.length,'index_A range error'
    
        item_info = json.loads(self.env[index])
        object_name = item_info['object']

        touch_dir = item_info['touch_dir'] # 000_0
        # touch A
        gelsightA_before_path = os.path.join(self.dataroot, touch_dir, 'gelsightA_before.jpg')
        gelsightA_after_path = os.path.join(self.dataroot, touch_dir, 'gelsightA_after.jpg')
        # touch B
        gelsightB_before_path = os.path.join(self.dataroot, touch_dir, 'gelsightB_before.jpg')
        gelsightB_after_path = os.path.join(self.dataroot, touch_dir, 'gelsightB_after.jpg')

        target = item_info['grasping']
        target = int(target)
        
        gelsightA_before = Image.open(gelsightA_before_path).convert('RGB')
        gelsightA_after = Image.open(gelsightA_after_path).convert('RGB')
        gelsightB_before = Image.open(gelsightB_before_path).convert('RGB')
        gelsightB_after = Image.open(gelsightB_after_path).convert('RGB')
        

        if self.transform is not None:
            gelsightA_before = self.transform(gelsightA_before)
            gelsightA_after = self.transform(gelsightA_after)
            gelsightB_before = self.transform(gelsightB_before)
            gelsightB_after = self.transform(gelsightB_after)

        out = torch.cat((gelsightA_before, gelsightA_after, gelsightB_before, gelsightB_after), dim=0)
    

        return out, target
    
    def __len__(self):
        """Return the total number of images."""
        return self.length


