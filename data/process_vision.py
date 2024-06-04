import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD



def get_vision_transform(args):
    # naive argumentation, still to be modified
    transform = transforms.Compose([  
        transforms.RandomRotation(degrees=(-20, 20)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop((224,224)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
    ])

    return transform

def load_and_transform_vision(vision_path, transform):
    vision = Image.open(vision_path)
    #Image.Image.show(vision)
    vision_outputs = transform(vision)
    #print(vision_outputs.shape)
    return {'pixel_values': vision_outputs}
