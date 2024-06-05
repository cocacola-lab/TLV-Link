import copy
import logging
import argparse
import os.path
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig
# import transformers
# transformers.logging.set_verbosity_error()



from model.base_model import TLVModel
from model.process_clip import add_time_attn_block, convert_model_to_lora, set_global_value, resize_pos
from open_clip import convert_weights_to_lp
from open_clip.transformer import PatchDropout
from training.distributed import is_master


def SET_GLOBAL_VALUE(k, v):
    set_global_value(k, v)

def create_vat_model(args):

    config = AutoConfig.from_pretrained(args.model)
    model= TLVModel(args, config, args.num_frames, args.add_time_attn, args.tube_size)
    model.touch_model.patch_dropout = PatchDropout(args.force_patch_dropout)
    model.vision_model.patch_dropout = PatchDropout(args.force_patch_dropout)

    device = args.device
    precision = args.precision
    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        convert_weights_to_lp(model, dtype=dtype)
    elif precision in ("pure_fp16", "pure_bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)

    if args.pretrained:
        try:
            args.pretrained = os.path.join(args.cache_dir, args.pretrained)
            ckpt = torch.load(args.pretrained, map_location='cpu')
            if args.clip_type == 'tlv':
                new_ckpt = {}
                for key,item in ckpt.items():
                    if "vision_model" in key:
                        new_ckpt[key] = item
                        new_ckpt[key.replace("vision_model","touch_model")] = copy.deepcopy(item)
                    elif "visual_projection" in key:
                        new_ckpt[key] = item
                        new_ckpt[key.replace("visual_projection", "touch_projection")] = copy.deepcopy(item)
                    elif "logit_scale" == key:
                        new_ckpt[key] = item
                        new_ckpt["tv_logit_scale"] = copy.deepcopy(item)
                        new_ckpt["vl_logit_scale"] = copy.deepcopy(item)
                    else:
                        new_ckpt[key] = item   
                incompatible_keys = model.load_state_dict(new_ckpt, strict=False if args.add_time_attn else True)
            else:
                incompatible_keys = model.load_state_dict(ckpt, strict=False if args.add_time_attn else True)

            # if is_master(args):
            #     logging.info(incompatible_keys)
        except Exception as e:
            if is_master(args):
                logging.info(f"Failed loading pretrained model with {e}")
    else:
        if is_master(args):
            logging.info(f"No pretrained model to load in \'{args.pretrained}\'")

    if args.add_time_attn:
        add_time_attn_block(model.touch_model.encoder, device=device)
        add_time_attn_block(model.vision_model.encoder, device=device)
        if is_master(args):
            logging.info(f'Convert spatial attention to time attention pretrained.')

    if args.init_temp != 0:
        with torch.no_grad():
            model.logit_scale.fill_(np.log(1 / float(args.init_temp)))
        if is_master(args):
            logging.info(f'Reset logit scale to {args.init_temp} (log-scale) and trainable {args.learn_temp}.')

    if args.convert_to_lora:
        convert_model_to_lora(args, model)
        if is_master(args):
            logging.info(f"Successfuly convert model to lora style.")

    return model