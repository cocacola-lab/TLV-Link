import contextlib
import io
import json
import logging
import os.path
import random
import re
import time

import pandas as pd

from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX

from data.process_text import load_and_transform_text
from data.process_touch import load_and_transform_touch, get_touch_transform
from data.process_vision import load_and_transform_vision, get_vision_transform



import argparse
from os.path import join as opj
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from PIL import Image
from torchvision import transforms
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


class TOUCH_dataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.sent_type = 'sentence_desc'
        self.phra_type = 'phrase_desc'

        self.touch100k = "/home/chenning/Projects/Touch100k/dataset/finetune/touch100k"
        self.data_root = self.touch100k

        self.id2title_folder_caps = []
        with open(args.train_data, 'r', encoding='utf-8') as f: 
            for line in f.readlines():
                item = json.loads(line)
                self.id2title_folder_caps.append(item)
        self.ids = self.id2title_folder_caps[:args.train_num_samples]


        self.tokenizer = get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir)
        self.vision_transform = get_vision_transform(args)
        self.touch_transform = get_touch_transform(args)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        try:
            sent_output, phra_output= self.get_text(idx)
            sent_input_ids, sent_attention_mask = sent_output['input_ids'], sent_output['attention_mask']
            phra_input_ids, phra_attention_mask = phra_output['input_ids'], phra_output['attention_mask']

            matched_modality_touch, matched_modality_vision = self.get_touch_vision(idx)
            return matched_modality_touch['pixel_values'], matched_modality_vision['pixel_values'], sent_input_ids, sent_attention_mask, phra_input_ids, phra_attention_mask        
        except Exception as error_msg:
            logging.info(f"Failed at {idx} with \"{error_msg}\"")
            return self.__getitem__(random.randint(0, self.__len__()-1))


    def get_text(self, id):
        # sentence-level
        sent = self.id2title_folder_caps[id][self.sent_type]
        sent_output = load_and_transform_text(sent, self.tokenizer)
        # phrase-level
        phra = self.id2title_folder_caps[id][self.phra_type]
        phra_output = load_and_transform_text(phra, self.tokenizer)
        return sent_output, phra_output
    
    def get_touch_vision(self, id):
        touch_folder = opj(self.data_root, 'touch')
        img = self.id2title_folder_caps[id]['img']
        touch_path = os.path.join(touch_folder, img)
        touch = load_and_transform_touch(touch_path, self.vision_transform)

        vision_folder = opj(self.data_root, 'vision')
        img = self.id2title_folder_caps[id]['img']
        vision_path = os.path.join(vision_folder, img)
        vision = load_and_transform_vision(vision_path, self.vision_transform)

        return touch,vision
    