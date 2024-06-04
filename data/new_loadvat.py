import ast
import io
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy.lib.format
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor
from tqdm import tqdm
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX
from training.params import parse_args
from data.process_text import load_and_transform_text
from data.process_video import get_video_transform
from data.process_audio import get_audio_transform
from data.process_depth import get_depth_transform
from data.process_thermal import get_thermal_transform
import pdb

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), \
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards



def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('raw.txt' in sample and 'mplug.txt' in sample and 'polish_mplug.txt' in sample and 'ofa3.txt' in sample)
    has_image = ('frm7.jpg' in sample and 'tml0.jpg' in sample and 'dep0.npy' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 200
_SHARD_SHUFFLE_INITIAL = 50
_SAMPLE_SHUFFLE_SIZE = 500
_SAMPLE_SHUFFLE_INITIAL = 100


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
            self,
            urls,
            weights=None,
            nshards=sys.maxsize,
            worker_seed=None,
            deterministic=False,
            epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), \
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


class Decode:
    def __init__(self, args=None):
        self.num_frames = args.num_frames
        self.text_type = args.text_type
        self.chatgpt = self.text_type == 'polish_mplug'
        self.title = self.text_type == 'raw'
        self.clip_type = args.clip_type
        self.tokenizer = get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir)
        self.video_transform = get_video_transform(args)
        self.audio_transform = get_audio_transform(args)
        self.depth_transform = get_depth_transform(args)
        self.thermal_transform = get_thermal_transform(args)


    def __call__(self, sample):
        input_ids, attention_mask = self.get_text(sample[f"{self.text_type}.txt"], chatgpt=self.chatgpt, title=self.title)
        if self.clip_type == 'vl':
            matched_modality = self.get_video([sample[f"frm{i}.jpg"] for i in range(self.num_frames)])
        elif self.clip_type == 'al':
            matched_modality = self.get_audio()
        elif self.clip_type == 'dl':
            matched_modality = self.get_depth(sample[f"dep0.npy"])
        elif self.clip_type == 'tl':
            matched_modality = self.get_thermal(sample[f"tml0.jpg"])
            # matched_modality = self.get_thermal(sample[f"tml{random.randint(0, 7)}.jpg"])
        else:
            raise ValueError
        return matched_modality, input_ids, attention_mask


    def get_video(self, frames):
        video_data = []
        for frame in frames:
            with io.BytesIO(frame) as stream:
                img = Image.open(stream)
                img.load()
            assert min(img.size) == 256
            result = ToTensor()(img)
            video_data.append(result)
        video_data = torch.stack(video_data, dim=1)
        # video_data torch.Size([3, 8, 455, 256])
        # video_outputs torch.Size([3, 8, 224, 224])
        video_outputs = self.video_transform(video_data)
        return video_outputs


    def get_text(self, text, chatgpt=True, title=False):
        text = text.decode("utf-8")
        if chatgpt:
            assert text.startswith('In the video, ')
            text = text[14:]
        tokens = load_and_transform_text(text, self.tokenizer, title=title)
        return tokens['input_ids'], tokens['attention_mask']

    def get_audio(self):
        raise NotImplementedError

    def get_depth(self, depth):
        stream = io.BytesIO(depth)
        img = numpy.lib.format.read_array(stream)
        depth = self.depth_transform(img)
        return depth

    def get_thermal(self, thermal):
        with io.BytesIO(thermal) as stream:
            img = Image.open(stream)
            img.load()
        thermal = self.thermal_transform(img)
        return thermal