import os
import json
import torch
import numpy as np
from sksurv.metrics import concordance_index_censored

"""
Contains metrics for survival tasks
"""

class SurvivalMixin:
    @staticmethod
    def survival_metrics(survival_events, survival_times, preds, saveto = None):
        """
        Calculate various survival metrics 
        
        Args:
            - survival_events (np.ndarray): All event indicators from test set. Shape (num_samples,)
            - survival_times (np.ndarray): All event times from test set. Shape (num_samples,)
            - preds (np.ndarray): Predicted risk scores from test set. Shape (num_samples,)
            
        Returns:
            - metrics (dict): Dictionary of metrics
        """
        # Convert survival_events to boolean
        survival_events = survival_events.astype(bool)
        
        # Assert shapes
        assert preds.ndim == 1, f"Predictions must be 1D, got shape {preds.shape}"
        num_samples = preds.shape[0]
        assert survival_events.shape == (num_samples,), f"Expected shape ({num_samples},) for survival_events, got {survival_events.shape}"
        assert survival_times.shape == (num_samples,), f"Expected shape ({num_samples},) for survival_times, got {survival_times.shape}"
        
        # Compute metrics
        metrics = {
            'cindex': concordance_index_censored(survival_events, survival_times, preds, tied_tol=1e-08)[0],
        }
        
        if saveto is not None:
            os.makedirs(os.path.dirname(saveto), exist_ok=True)
            with open(saveto, 'w') as f:
                json.dump(metrics, f, indent=4)

        return metrics

    @staticmethod
    def _calculate_risk(logits):
        """
        Take the logits of the model and calculate the risk for the patient.
        Adapted from: https://github.com/mahmoodlab/SurvPath/blob/fe4a97bf8fc57925dc81ff930ef7e1d9b2bbc83a/utils/core_utils.py#L409
        
        Args: 
            - logits (torch.Tensor): Logits returned by the model. Shape (num_samples,)
        
        Returns:
            - risk (np.array) : Risk for the patient, scalar.
        
        """
        hazards = torch.sigmoid(logits) # Shape (num_samples,)
        survival = torch.cumprod(1 - hazards, dim=0) # Shape (num_samples,)
        risk = -torch.sum(survival, dim=0) # Shape (num_samples,)
        return risk