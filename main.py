import glob
import logging
import os
import sys

import re
import subprocess
import sys
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from transformers import CLIPPreTrainedModel
# import transformers
# transformers.logging.set_verbosity_error()
from zero_shot.zeroshot_cls import evaluate_t_cls
from model.process_clip import set_global_value, print_trainable_parameters

try:
    import wandb
except ImportError:
    wandb = None

try:
    import tensorboardX as tensorboard
except ImportError:
    tensorboard = None

from data.build_datasets import get_data
from open_clip import create_model_and_transforms, create_loss
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.file_utils import pt_load, start_sync_process, remote_sync
from train import train_one_epoch
from model.build_model import create_vat_model

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
MODEL_DICT = {"ViT-L-14": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"}
CHECKPOINT_DICT = {"ViT-L-14": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/pytorch_model.bin"}


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None

def SET_GLOBAL_VALUE(k, v):
    set_global_value(k, v)

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%m_%d-%H_%M")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '_'.join([
            date_str,
            f"B{args.beta_init}_span{args.span}_{args.inte_type[:3]}_{args.decay_type[:3]}",
            f"{args.lr}-{args.batch_size}-{args.epochs}-{args.lr_scheduler}", 
        ])
    args.pretrained = CHECKPOINT_DICT[args.model]
    args.model = MODEL_DICT[args.model]
    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_base_path = log_base_path
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        # FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        # FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)

    model = create_vat_model(args)
    args.touch_image_size = model.touch_model.config.image_size
    args.vision_image_size = model.vision_model.config.image_size

    if args.distill:
        # FIXME: currenlty assumes the model your distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model,
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
        )
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    # if args.trace:
    #     model = trace_model(model, batch_size=args.batch_size, device=device)
    if args.lock_image:
        for param in model.touch_model.embeddings.parameters():
            param.requires_grad = False
        for param in model.touch_model.pre_layrnorm.parameters():
            param.requires_grad = False
        for param in model.vision_model.embeddings.parameters():
            param.requires_grad = False
        for param in model.vision_model.pre_layrnorm.parameters():
            param.requires_grad = False


    if not args.convert_to_lora:
        for param in model.touch_model.embeddings.parameters():
            param.requires_grad = False
        for param in model.touch_model.pre_layrnorm.parameters():
            param.requires_grad = False
        for param in model.vision_model.embeddings.parameters():
            param.requires_grad = False
        for param in model.vision_model.pre_layrnorm.parameters():
            param.requires_grad = False
        if args.add_time_attn:
            for name, param in model.touch_model.encoder.layers.named_parameters():
                if 'temporal' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for name, param in model.vision_model.encoder.layers.named_parameters():
                if 'temporal' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for name, param in model.touch_model.encoder.layers.named_parameters():
                if 'self_attn' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for name, param in model.vision_model.encoder.layers.named_parameters():
                if 'self_attn' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    else:
        if args.add_time_attn:
            for name, param in model.touch_model.encoder.layers.named_parameters():
                if 'temporal_embedding' in name or 'temporal_layer_norm1' in name:
                    param.requires_grad = True
            for name, param in model.vision_model.encoder.layers.named_parameters():
                if 'temporal_embedding' in name or 'temporal_layer_norm1' in name:
                    param.requires_grad = True

    for param in model.touch_model.embeddings.position_embedding.parameters():
        param.requires_grad = False
    model.touch_model.embeddings.class_embedding.requires_grad = True
    for param in model.vision_model.embeddings.position_embedding.parameters():
        param.requires_grad = False
    model.vision_model.embeddings.class_embedding.requires_grad = True

    if args.lock_text:
        for param in model.text_model.parameters():
            param.requires_grad = False
        for param in model.text_projection.parameters():
            param.requires_grad = False


    model.logit_scale.requires_grad = args.learn_temp

    if is_master(args):
        print_trainable_parameters(model, msg='The model: ')

    if args.grad_checkpointing:
        # model.text_model.encoder.gradient_checkpointing = args.grad_checkpointing
        model.touch_model.encoder.gradient_checkpointing = args.grad_checkpointing
        model.vision_model.encoder.gradient_checkpointing = args.grad_checkpointing
        


    if is_master(args):
        args_file = os.path.join(args.logs, args.name, "args.txt")
        with open(args_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                f.write(f"{name}: {val}\n")

    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args, find_unused_parameters=True)

        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    ############################################################
    # if args.train_data or args.dataset_type == "synthetic":
    assert not args.trace, 'Cannot train with traced model'

    no_decay = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n or 'class_embedding' in n or 'patch_embedding' in n
    decay = lambda n, p: not no_decay(n, p)

    lora = lambda n, p: "lora" in n
    non_lora = lambda n, p: not lora(n, p)

    named_parameters = list(model.named_parameters())
    no_decay_non_lora_params = [[n, p] for n, p in named_parameters if no_decay(n, p) and non_lora(n, p) and p.requires_grad]
    decay_non_lora_params = [[n, p] for n, p in named_parameters if decay(n, p) and non_lora(n, p) and p.requires_grad]

    no_decay_lora_params = [[n, p] for n, p in named_parameters if no_decay(n, p) and lora(n, p) and p.requires_grad]
    decay_lora_params = [[n, p] for n, p in named_parameters if decay(n, p) and lora(n, p) and p.requires_grad]


    param_groups = []
    if no_decay_non_lora_params: param_groups.append({"params": [p for n, p in no_decay_non_lora_params], "weight_decay": 0., 'lr': args.lr * args.coef_lr})
    if decay_non_lora_params: param_groups.append({"params": [p for n, p in decay_non_lora_params], "weight_decay": args.wd, 'lr': args.lr * args.coef_lr})
    if no_decay_lora_params: param_groups.append({"params": [p for n, p in no_decay_lora_params], "weight_decay": 0.})
    if decay_lora_params: param_groups.append({"params": [p for n, p in decay_lora_params], "weight_decay": args.wd})

    optimizer = optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    name_groups = {}
    if no_decay_non_lora_params:
        name_groups['no_decay_non_lora_params'] = [{"name": n, "weight_decay": 0., 'lr': args.lr * args.coef_lr} for n, p in no_decay_non_lora_params]
    if decay_non_lora_params:
        name_groups['decay_non_lora_params'] = [{"name": n, "weight_decay": args.wd, 'lr': args.lr * args.coef_lr} for n, p in decay_non_lora_params]
    if no_decay_lora_params:
        name_groups['no_decay_lora_params'] = [{"name": n, "weight_decay": 0., 'lr': args.lr} for n, p in no_decay_lora_params]
    if decay_lora_params:
        name_groups['decay_lora_params'] = [{"name": n, "weight_decay": args.wd, 'lr': args.lr} for n, p in decay_lora_params]
    if is_master(args):
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for group_name, group in name_groups.items():
                f.write(f"Group name: {group_name}:\n")
                for i in group:
                    f.write(f"Parameter name: {i['name']}. Learning rate: {i['lr']}. Weight decay: {i['weight_decay']}\n")


    scaler = GradScaler() if args.precision == "amp" else None
    ############################################################

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
        
            if optimizer is not None:
                if args.do_train:
                    optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    data = get_data(args, epoch=start_epoch)
    if is_master(args):
        logging.info(f"{data})")
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if f'{args.clip_type}_pt' in data and optimizer is not None:
        total_steps = (data[f'{args.clip_type}_pt'].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, \
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data[f'{args.clip_type}_pt'].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    # if args.wandb and is_master(args):
    #     assert wandb is not None, 'Please install wandb.'
    #     logging.debug('Starting wandb.')
    #     args.train_sz = data["train"].dataloader.num_samples
    #     if args.val_data is not None:
    #         args.val_sz = data["val"].dataloader.num_samples
    #     # you will have to configure this for your project!
    #     wandb.init(
    #         project=args.wandb_project_name,
    #         name=args.name,
    #         id=args.name,
    #         notes=args.wandb_notes,
    #         tags=[],
    #         resume='auto' if args.resume == "latest" else None,
    #         config=vars(args),
    #     )
    #     if args.debug:
    #         wandb.watch(model, log='all')
    #     wandb.save(params_file)
    #     logging.debug('Finished loading wandb.')

    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(model)

    if f'{args.clip_type}_pt' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        
        if "t_cls" in data:
            for sub_data in data['t_cls']:
                evaluate_t_cls(model, sub_data, start_epoch, args, writer)
        return

    torch.distributed.barrier()
    loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        if "t_cls" in data:
            for sub_data in data['t_cls']:
                metrics = evaluate_t_cls(model, sub_data, completed_epoch, args, writer)
        
        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                    args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)
    
    torch.distributed.barrier()
    
    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')





if __name__ == "__main__":
    main(sys.argv[1:])
