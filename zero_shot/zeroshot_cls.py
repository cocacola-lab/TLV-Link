
import json
import logging
import os

import torch.distributed
from training.distributed import is_master
from .zero_shot import zero_shot_eval
import torch

try:
    import wandb
except ImportError:
    wandb = None



def evaluate_t_cls(model, data, epoch, args, tb_writer=None):
    metrics = {}
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    if not metrics:
        return metrics
    
    if torch.distributed.get_rank() == 0:
        logging.info(
            f"Eval Epoch: {epoch-1} "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"material/t_cls/{args.val_t_cls_data[0].lower()}/{name}", val, epoch)
        args.t_cls_output_dir = os.path.join(args.log_base_path, f't_cls/{args.val_t_cls_data[0].lower()}')
        os.makedirs(args.t_cls_output_dir, exist_ok=True)
        with open(os.path.join(args.t_cls_output_dir, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})
    
    return metrics
