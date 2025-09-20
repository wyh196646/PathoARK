import math
from torch.optim import Optimizer

"""
Optimization utilities from GigaPath (https://github.com/prov-gigapath/prov-gigapath).
"""

class CustomLRScheduler:
    def __init__(self, optimizer: Optimizer,
                 default_scheduler_args: dict,
                 custom_scheduler_args: dict):
        '''
        Decay the learning rate with half-cycle cosine after warmup. Designed to be stepped on every accumulation_steps batches.
        Used by GigaPath, Threads, and Chief models.
        Originally based on https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
        
        Args:
            optimizer (Optimizer): Optimizer to adjust learning rate for
            default_scheduler_args (dict): Default scheduler configuration dictionary with the following keys:
                base_lr (float): Base learning rate (from optimizer)
                max_epochs (int): Total number of epochs
                accumulation_steps (int): Number of batches to accumulate gradients over
                len_dataloader (int): Number of batches per training epoch
            custom_scheduler_args (dict): Configuration dictionary with the following keys:
                warmup_epochs (int): Number of epochs for linear warmup
                min_lr (float): Minimum learning rate (reached at end)
                step_on (str): 'batch' or 'epoch'
        '''
        assert custom_scheduler_args['step_on'] == 'accumulation-step', f'This scheduler is designed to be stepped on each accumulation-step, not {custom_scheduler_args["step_on"]}'
        
        self.optimizer = optimizer
        self.base_lr = default_scheduler_args["base_lr"]
        self.max_iters = default_scheduler_args["max_epochs"] * default_scheduler_args["len_dataloader"]
        self.warmup_iters = custom_scheduler_args["warmup_epochs"] * default_scheduler_args["len_dataloader"]
        self.min_lr = custom_scheduler_args["min_lr"]

        self.current_iter = 0 # Initialize progress tracker
        self.step(0) # Initialize learning rate

    def get_lr(self) -> float:
        if self.current_iter < self.warmup_iters:
            lr = self.base_lr * self.current_iter / self.warmup_iters
        else:
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (self.current_iter - self.warmup_iters) / (self.max_iters - self.warmup_iters)))
        return lr

    def step(self, total_progress) -> float:
        self.current_iter = total_progress * self.max_iters
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
    
    def get_last_lr(self) -> list:
        '''
        Returns a list of the last computed learning rate for each parameter group.
        '''
        try:
            return [group['lr'] for group in self.optimizer.param_groups if group['lr_scale'] == 1] # Only gigapath model uses lr_scale for now
        except:
            return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            'base_lr': self.base_lr,
            'max_iters': self.max_iters,
            'warmup_iters': self.warmup_iters,
            'min_lr': self.min_lr,
            'current_iter': self.current_iter
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        self.base_lr = state_dict['base_lr']
        self.max_iters = state_dict['max_iters']
        self.warmup_iters = state_dict['warmup_iters']
        self.min_lr = state_dict['min_lr']
        self.current_iter = state_dict['current_iter']
        



def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    
    """
    A function for getting custom parameter groups, as implemented in the official Gigapath codebase.
    Each layer has a custom learning rate scale, and the mask token and decoder parameters are excluded from training.
    Used for Gigapath finetuning.

    # Modified from https://github.com/prov-gigapath/prov-gigapath/blob/11e8b477c005060a18653fdf0c53612186d3f117/finetune/utils.py#L209C1-L255C53
    """
    # ------------------------------------------------------------------------------------------
    # References:
    # BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    # ------------------------------------------------------------------------------------------
    param_group_names = {}
    param_groups = {}

    # num_layers = model.slide_encoder.encoder.num_layers + 1
    num_layers = model.slide_encoder.model.encoder.num_layers + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        # if 'mask_token' in n or 'slide_encoder.decoder' in n:
        if 'mask_token' in n or 'slide_encoder.model.decoder' in n:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
        
        layer_id = get_layer_id(n, num_layers)

        group_name = n + "_%d_%s" % (layer_id + 1, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id(name, num_layers):
    # ------------------------------------------------------------------------------------------
    # References:
    # BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    # ------------------------------------------------------------------------------------------
    if 'cls_token' in name or 'pos_embed' in name:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    # elif name.startswith('slide_encoder.encoder.layers'):
    elif name.startswith('slide_encoder.model.encoder.layers'):
        return int(name.split('.')[4]) + 1
    else:
        return num_layers