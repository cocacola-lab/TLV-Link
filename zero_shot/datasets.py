import os
import cv2
import torch
import json

from data.build_datasets import DataInfo
from data.process_touch import get_touch_transform
from torchvision import datasets,transforms
from torch.utils.data import Dataset
from PIL import Image


class FeelingLabel(Dataset):
    """datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False, mode='train', label='full', data_amount=100):
        self.two_crop = two_crop
        self.dataroot = 'dataset/downstream/feeling/data'
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
        gelsightA_during_path = os.path.join(self.dataroot, touch_dir, 'gelsightA_during.jpg')
        gelsightA_after_path = os.path.join(self.dataroot, touch_dir, 'gelsightA_after.jpg')
        # touch B
        gelsightB_before_path = os.path.join(self.dataroot, touch_dir, 'gelsightB_before.jpg')
        gelsightB_during_path = os.path.join(self.dataroot, touch_dir, 'gelsightB_during.jpg')
        gelsightB_after_path = os.path.join(self.dataroot, touch_dir, 'gelsightB_after.jpg')

        target = item_info['grasping']
        target = int(target)
        
        gelsightA_before = Image.open(gelsightA_before_path).convert('RGB')
        gelsightA_during = Image.open(gelsightA_during_path).convert('RGB')
        gelsightA_after = Image.open(gelsightA_after_path).convert('RGB')
        gelsightB_before = Image.open(gelsightB_before_path).convert('RGB')
        gelsightB_during = Image.open(gelsightB_during_path).convert('RGB')
        gelsightB_after = Image.open(gelsightB_after_path).convert('RGB')
        

        if self.transform is not None:
            gelsightA_before = self.transform(gelsightA_before)
            gelsightA_during = self.transform(gelsightA_during)
            gelsightA_after = self.transform(gelsightA_after)
            gelsightB_before = self.transform(gelsightB_before)
            gelsightB_during = self.transform(gelsightB_during)
            gelsightB_after = self.transform(gelsightB_after)

        out = torch.cat((gelsightA_before, gelsightA_after, gelsightB_before, gelsightB_after), dim=0)
        out_during = torch.cat((gelsightA_during,gelsightB_during), dim=0)
    

        return out, target
    
    def __len__(self):
        """Return the total number of images."""
        return self.length

def get_feeling_dataset(args):
    data_folder = args.touch_data_path

    mean= [0.485, 0.456, 0.406]
    std= [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
      
    dataset = FeelingLabel(data_folder, transform=transform, mode='test')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=None,
    )

    return DataInfo(dataloader=dataloader, sampler=None)



def get_tag_dataset(args):
    data_path = args.touch_data_path
    transform = get_touch_transform(args)
    dataset = datasets.ImageFolder(data_path, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=None,
    )

    return DataInfo(dataloader=dataloader, sampler=None)
