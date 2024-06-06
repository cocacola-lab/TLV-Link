import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
import open_clip



def get_touch_transform(args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(224),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)  # assume image
        ]
    )

    
    return transform

def load_and_transform_touch(touch_path, transform):
    touch = Image.open(touch_path)
    touch_outputs = transform(touch)
    return {'pixel_values': touch_outputs}
