import json
import os
import matplotlib.pyplot as plt
from patho_bench.config.JSONSaver import JSONsaver
import torch
import torch.nn as nn

"""
This file contains the LoggingMixin class which provides methods for logging metrics and saving checkpoints during training.
Classes:
    LoggingMixin: A mixin class for logging metrics and saving checkpoints during training.
Methods:
    compute_extra_metrics(): OPTIONAL. Compute extra metrics based on self.current_epoch_metrics['outputs'] and add to self.current_epoch_metrics.
    init_loggers(): OPTIONAL. Initialize metrics loggers.
    log_lr(step): Log learning rate to dashboard.
    log_loss(step): Log loss to dashboard.
    log_smooth_rank(step): Log smooth rank to dashboard.
    save_current_epoch_metrics(saveto): Saves self.current_epoch_metrics to disk.
    log_progress(): Logs progress to progress.txt file in the experiment directory.
    do_callbacks(mode): Runs callbacks for the current epoch.
    save_checkpoints(): Saves checkpoint.
    get_checkpoint_path_for_current_fold(how): Gets the desired checkpoint path for the current fold.
"""


class LoggingMixin:
    def compute_extra_metrics(self):
        '''
        OPTIONAL. Compute extra metrics based on self.current_epoch_metrics['outputs'] and add to self.current_epoch_metrics

        Sets:
            self.current_epoch_metrics (dict): Dictionary of metrics for this epoch {'loss': loss, 'outputs': outputs, 'extra_metric_1': extra_metric_1, ...}
        '''
        pass

    def init_loggers(self, save_dir):
        '''
        OPTIONAL. Initialize metrics loggers. This method can be overwritten in child classes to use different logger(s).

        Sets:
            self.logger (Any): Logger object
        '''
        return {'loss': TrainingMetricsLogger(save_dir, 'loss', step_on = 'epoch'),
                'lr': TrainingMetricsLogger(save_dir, 'lr', step_on = 'batch'),
                'smooth_rank': TrainingMetricsLogger(save_dir, 'smooth_rank', step_on = 'epoch')}
        
    def log_lr(self, step):
        '''
        Log learning rate to dashboard. This method can be overwritten in child classes to log learning rate differently.

        You may find the following attributes useful:
            self.current_epoch (int): Current epoch idx
            self.mode (str): Mode of operation, either 'train', 'val', or 'test'.
            self.scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        '''
        self.loggers['lr'].step({self.mode: self.scheduler.get_last_lr()[0]}, step)

    def log_loss(self, step):
        '''
        Log loss to dashboard. This method can be overwritten in child classes to log loss differently.

        You may find the following attributes useful:
            self.current_epoch (int): Current epoch idx
            self.current_epoch_metrics (dict): Dictionary of metrics for this epoch {'loss': loss, 'outputs': outputs, 'extra_metric_1': extra_metric_1, ...}
            self.mode (str): Mode of operation, either 'train', 'val', or '
        '''
        self.loggers['loss'].step({self.mode: self.current_epoch_metrics['per_sample_loss']}, step)

    def log_smooth_rank(self, step):
        '''
        Log smooth rank to dashboard. This method can be overwritten in child classes to log smooth rank differently.

        You may find the following attributes useful:
            self.current_epoch (int): Current epoch idx
            self.current_epoch_metrics (dict): Dictionary of metrics for this epoch {'loss': loss, 'outputs': outputs, 'extra_metric_1': extra_metric_1, ...}
            self.mode (str): Mode of operation, either 'train', 'val', or 'test'.
        '''
        if 'smooth_rank' in self.current_epoch_metrics:
            self.loggers['smooth_rank'].step({self.mode: self.current_epoch_metrics['smooth_rank']}, step)

    def save_current_epoch_metrics(self, save_path):
        '''
        Saves self.current_epoch_metrics to disk.

        Args:
            save_path (str): Path to save metrics.
        '''
        # Save metrics to disk
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.current_epoch_metrics, f, cls=JSONsaver, indent=4)

        # Reset epoch metrics
        self.current_epoch_metrics = {}
                        
    @staticmethod
    def save_model_architecture(model: nn.Module, save_path: str):
        '''
        Saves model architecture to disk, along with parameter counts
        
        Args:
            model (nn.Module): Model to save.
            save_path (str): Path to save model architecture.
        '''
        import patho_bench.experiments.utils.FancyLayers # Overrides the default __repr__ method for all torch.nn.Module objects so that it shows frozen and trainable layers
        
        with open(save_path, "w") as f:
            f.write(str(model))
            f.write("\n")
            f.write(f"Total number of parameters: {sum(param.numel() for param in model.parameters())} \n")
            f.write(f"Total number of trainable parameters: {sum(param.numel() for param in model.parameters() if param.requires_grad)} \n")

    @staticmethod
    def save_checkpoint(model: nn.Module, method: str, save_path: str):
        '''
        Saves current checkpoint. If method is 'best-val-loss' or 'best-smooth-rank', deletes previous best checkpoint.
        
        Args:
            model (nn.Module): Model to save.
            method (str): Method to save checkpoint. One of ['best-val-loss', 'best-smooth-rank', 'last']
            save_path (str): Path to save checkpoint.
        '''
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        # Delete previous best checkpoint
        if method in ['best-val-loss', 'best-smooth-rank']:
            existing_checkpoints = [os.path.join(save_dir, checkpoint) for checkpoint in os.listdir(save_dir)]
            assert len(existing_checkpoints) <=1, f"More than one checkpoint found in {save_dir}, even though only one is expected due to save method {method}."
            if len(existing_checkpoints) > 0:
                os.remove(existing_checkpoints[0])
        
        # Save current checkpoint
        torch.save(model.state_dict(), save_path)
        
    @staticmethod
    def load_checkpoint(model: nn.Module, load_from: str, verbose = False):
        '''
        Loads a checkpoint into the model.

        Args:
            model (nn.Module): Model to load checkpoint into.
            load_from (str): Path to checkpoint.
        '''       
        assert os.path.exists(load_from), f'Checkpoint {load_from} not found'
        ckpt_clean = torch.load(load_from, map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt_clean, strict=True)
        if verbose:
            print(f'\033[92mLoaded checkpoint from {load_from}\033[0m')
        return model
    
    @staticmethod
    def freeze(model, layers = None):
        '''
        Freezes parameters and set to eval for given layers of the model

        Args:
            layers (list): List of layer names to freeze. If None, freezes and evals all layers.
        '''
        assert layers is None or isinstance(layers, list), f'layers must be a list or None, received {layers} instead'
        if layers is None:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            model.frozen = 'all'
        else:
            for layer in layers:
                for name, param in model.named_parameters():
                    if name.startswith(layer):
                        # print(f'Freezing {name} child of {layer}')
                        param.requires_grad = False
                for name, module in model.named_modules():
                    if name.startswith(layer):
                        module.eval()
            model.frozen = layers
            
        return model
            
            


class TrainingMetricsLogger:
    '''
    Custom logging class for logging and visualizing training metrics.
    
    Parameters
    ----------
    save_dir (str): The directory to save the dashboard images to.
    name (str): The name of the dashboard. Will be saved as "name.png".
    step_on (str): What each step represents. Usually "epoch" or "batch", but can be anything. Used for x-axis label.
    '''
    def __init__(self, save_dir, name, step_on):
        self.save_dir = save_dir
        self.name = name
        self.step_on = step_on
        self.data = {}
        self.current_step = {} # Initialize step counter for each line

    def step(self, metrics, step, save = True, show = False):
        '''
        Logs the metrics data and saves an updated dashboard image. Previous dashboard images are overwritten.

        Parameters
        ----------
        metrics : dict
            A dictionary where every key corresponds to a line in this plot labeled by the second key.
        step : int
            The step number.
        save : bool, optional
            Whether to save the dashboard on this step. The default is True.
        show : bool, optional
            Whether to show the dashboard on this step. The default is False.
        '''        
        # Add new data
        for line_name, y in metrics.items():
            if y is not None:
                if line_name not in self.data:
                    self.data[line_name] = []
                self.data[line_name].append((step, y))

        # Create subplots based on the number of metrics
        fig, ax = plt.subplots(figsize=(15, 5))

        # Plot updated data
        for line_name, points in self.data.items():
            points.sort(key=lambda x: x[0])  # sort by x
            x, y = zip(*points)  # unzip into two lists
            ax.plot(x, y, label=line_name, marker="o")
            ax.legend()
            ax.set_title(self.name)
            ax.set_facecolor("white")  # set plot background to white
            ax.set_xlabel(self.step_on)
            if step < 10:
                ax.set_xticks(range(step+1))
            elif step < 50:
                ax.set_xticks(list(range(0, step+1, 5)) + [step])
            elif step < 100:
                ax.set_xticks(list(range(0, step+1, 10)) + [step])
            elif step < 500:
                ax.set_xticks(list(range(0, step+1, 50)) + [step])  
            else:
                ax.set_xticks(list(range(0, step+1, 100)) + [step])

        # Set figure background and layout
        fig.set_facecolor("white")
        plt.tight_layout()

        # Display and save logic
        if show:
            plt.show()
        if save:
            os.makedirs(self.save_dir, exist_ok=True)
            plt.savefig(os.path.join(self.save_dir, f'{self.name}.png'))
        plt.close()

    def load(self, filename):
        '''
        Loads file from self.save_dir and returns the data.
        '''
        with open(os.path.join(self.save_dir, filename), "r") as f:
            return json.load(f)