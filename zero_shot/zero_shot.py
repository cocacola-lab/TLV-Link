import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX
from .precision import get_autocast
from .zero_shot_classifier import build_zero_shot_classifier
from .zero_shot_metadata import CLASSNAMES, NUM_CLASSES, OPENAI_IMAGENET_TEMPLATES


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    embedds_list = []
    labels_list = []
    
    with torch.no_grad():
        top1, top2, n = 0., 0., 0.
        preds, labels = [], []
        for images, target in tqdm(dataloader, total=len(dataloader)) if args.local_rank == 0 else dataloader:
            if args.cls_mode == 'grasping':

                # --------before and after
                gelsightA_before = images[:,:3,:,:]
                gelsightA_after = images[:,3:6,:,:]
                gelsightB_before = images[:,6:9,:,:]
                gelsightB_after = images[:,9:12,:,:]

                gelsightA_before = gelsightA_before.to(device=args.local_rank, dtype=input_dtype)
                gelsightA_after = gelsightA_after.to(device=args.local_rank, dtype=input_dtype)
                gelsightB_before = gelsightB_before.to(device=args.local_rank, dtype=input_dtype)
                gelsightB_after = gelsightB_after.to(device=args.local_rank, dtype=input_dtype)


                #-----------target
                target = target.to(args.local_rank)

                with autocast():
                    
                    # predict
                    gelsightA_before_output = model(args=args, touch=gelsightA_before)
                    gelsightA_after_output = model(args=args, touch=gelsightA_after)
                    gelsightB_before_output = model(args=args, touch=gelsightB_before)
                    gelsightB_after_output = model(args=args, touch=gelsightB_after)


                    gelsightA_before_features = gelsightA_before_output['touch_features'] if isinstance(gelsightA_before_output, dict) else gelsightA_before_output[0]
                    gelsightA_after_features = gelsightA_after_output['touch_features'] if isinstance(gelsightA_after_output, dict) else gelsightA_after_output[0]
                    gelsightB_before_features = gelsightB_before_output['touch_features'] if isinstance(gelsightB_before_output, dict) else gelsightB_before_output[0]
                    gelsightB_after_features = gelsightB_after_output['touch_features'] if isinstance(gelsightB_after_output, dict) else gelsightB_after_output[0]

                    

                    # logits = 100. * gelsightA_during_features @ classifier
                    gelsightA_before_logits = 100. * gelsightA_before_features @ classifier
                    gelsightA_after_logits = 100. * gelsightA_after_features @ classifier
                    gelsightB_before_logits = 100. * gelsightB_before_features @ classifier
                    gelsightB_after_logits = 100. * gelsightB_after_features @ classifier
                    # logits = gelsightA_before_logits * 0.5 + gelsightA_after_logits * 0.5
                    logits = gelsightA_before_logits * 0.25 + gelsightA_after_logits * 0.25 + gelsightB_before_logits * 0.25 + gelsightB_after_logits * 0.25
                    image_features = gelsightA_before_features * 0.25 + gelsightA_after_features * 0.25 + gelsightB_before_features * 0.25 + gelsightB_before_features* 0.25
                    # gelsight = 0.5 * gelsightA_before_features + 0.5 * gelsightA_after_features
                    # logits =  100. * gelsightA_during_features @ classifier

            else:
                images = images.to(device=args.local_rank, dtype=input_dtype)
                images = images.unsqueeze(2)
                target = target.to(args.local_rank)

                with autocast():
                    # predict
                    output = model(args=args, touch=images)
                    image_features = output['touch_features'] if isinstance(output, dict) else output[0]
                    logits = 100. * image_features @ classifier
                    
            embedds_list.append(image_features.cpu().numpy())
            labels_list.append(target.cpu().numpy())

            preds.append(torch.softmax(logits, axis=1)[:, 1].cpu().numpy())
            labels.append(target.cpu().numpy())
            # measure accuracy
            #print(len(logits), logits.sum())
            acc1, acc2 = accuracy(logits, target, topk=(1, 2))
            top1 += acc1
            top2 += acc2
            n += images.size(0)

    top1 = (top1 / n)
    top2 = (top2 / n)

    return top1, top2


def zero_shot_eval(model, data, epoch, args):
    temp_val_t_cls_data = args.val_t_cls_data
    args.val_t_cls_data = list(data.keys())
    assert len(args.val_t_cls_data) == 1
    args.val_t_cls_data = args.val_t_cls_data[0]

    if args.val_t_cls_data not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module
        
    if args.local_rank == 0:
        logging.info(f'Starting zero-shot {args.val_t_cls_data.upper()}.')

    if args.local_rank == 0:
        logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    with autocast():
        tokenizer = get_tokenizer(HF_HUB_PREFIX+args.model, cache_dir=args.cache_dir)
        # tokenizer = get_tokenizer("ViT-L-14")
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=CLASSNAMES[args.val_t_cls_data][args.cls_mode],
            templates=OPENAI_IMAGENET_TEMPLATES[args.cls_mode],
            num_classes_per_batch=NUM_CLASSES[args.val_t_cls_data][args.cls_mode],
            device=args.device,
            use_tqdm=False,
        )

    if args.local_rank == 0:
        logging.info('Using classifier')
    results = {}
    if args.val_t_cls_data in data:
        top1, top2= run(model, classifier, data[args.val_t_cls_data].dataloader, args)
        results[f'{args.val_t_cls_data}-zeroshot-val-top1'] = top1
        results[f'{args.val_t_cls_data}-zeroshot-val-top2'] = top2


    if args.local_rank == 0:
        logging.info(f'Finished zero-shot {args.val_t_cls_data.upper()}.')

    args.val_t_cls_data = temp_val_t_cls_data
    return results
