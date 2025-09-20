import copy
from torch import nn
from patho_bench.optim.NLLSurvLoss import NLLSurvLoss
from patho_bench.Pooler import Pooler

"""
This is a wrapper class that allows for finetuning of a slide encoder model on a single multiple-instance learning (MIL) classification task.
This class is used by ExperimentFactory.
"""

class TrainableSlideEncoder(nn.Module):
    def __init__(self,
                 slide_encoder,
                 post_pooling_dim,
                 task_name,
                 num_classes,
                 loss,
                 device):
        '''
        Initializes a trainable classifier using a preloaded slide encoder.

        Args:
            slide_encoder (nn.Module, optional): The image pooling module used to process input features. Input shape: batch_size x num_patches x feature_dim. Output shape: batch_size x post_pooling_dim.
            post_pooling_dim (int, optional): Dimension of features after pooling.
            task_name (str): Name of the task.
            num_classes (int): The number of classes.
            loss (nn.Module or dict): Loss function to use for training. If a dictionary is provided, it should map task names to loss functions.
            device (str or torch.device, optional): Device on which to run the model.
        '''
        super().__init__()
        self.slide_encoder = copy.deepcopy(slide_encoder) # Deepcopy so that the original slide encoder is not modified across folds
        self.post_pooling_dim = post_pooling_dim
        self.task_name = task_name
        self.num_classes = num_classes
        self.loss = loss
        self.device = device

        # Create classification head
        # Input shape: batch_size x feature_dim
        # Output shape: batch_size x num_classes
        self.classification_head = nn.Linear(self.post_pooling_dim, self.num_classes)

        # Move to device
        self.to(device)
        if isinstance(self.loss, dict): # If balanced loss is used
            for iter_idx, loss in self.loss.items():
                self.loss[iter_idx].to(device)
        else:
            self.loss.to(device)

    def forward(self, batch, output = 'loss'):
        '''        
        Args:
            batch (dict): Input batch containing 'slide' and 'labels' keys.
            output (str): 'loss', 'features', or 'logits'
        Returns:
            Logits (shape: batch_size x n_categories) if return_loss is False, otherwise loss and accuracy
        '''
        # Slide encoding
        slide_encoder_input = Pooler.prepare_slide_encoder_input_batch(batch['slide'])
        slide_features = Pooler.pool(self.slide_encoder, slide_encoder_input, self.device)
        
        if output == 'features':
            return slide_features
        
        # Classification heads
        logits = self.classification_head(slide_features)
        if output == 'logits':
            return logits
        
        # Compute survival loss
        if isinstance(self.loss, NLLSurvLoss):
            # Note that survival task labels follow a particular format. If the expected format is unclear from the examples, please raise a GitHub issue.
            y_bins = batch['labels'][self.task_name].to(self.device)
            y_bins = y_bins % 4 # Convert y_bins from 8 to 4 bins (survival quartiles)
            y_event = batch['labels']['extra_attrs'][f'{self.task_name}_event'].to(self.device)
            loss = self.loss(logits, y_bins.unsqueeze(0), y_event.unsqueeze(0))
        # Compute balanced loss
        elif isinstance(self.loss, dict):
            assert batch.get('current_iter') is not None, "Current iter must be provided for weighted loss, but got None from batch. Please check the dataloader."
            loss = self.loss[batch['current_iter']](logits.squeeze(), batch['labels'][self.task_name].to(self.device).squeeze())
        # Compute standard loss
        else:
            loss = self.loss(logits.squeeze(), batch['labels'][self.task_name].to(self.device).squeeze())

        if output == 'loss':
            info = [{}]
            return loss, info
        
        raise ValueError(f"Invalid output type {output}")