import cv2
import torch

from data.build_datasets import DataInfo
from data.process_vision import get_vision_transform
from torchvision import datasets

def get_vision_dataset(args):
    data_path = args.vision_data_path
    transform = get_vision_transform(args)
    dataset = datasets.ImageFolder(data_path, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=None,
    )

    return DataInfo(dataloader=dataloader, sampler=None)
