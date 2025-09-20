import numpy as np
import seaborn as sns
import os
import json
import torch
sns.set_style("white")

"""
Contains visualization methods for classification tasks
"""

class RetrievalMixin:
    def retrieval_metrics(self, labels, preds, ks, saveto=None):
        """
        Compute retrieval metrics, both overall and per class.

        Args:
            labels (torch.Tensor): True labels for queries (shape: n).
            preds (torch.Tensor): Labels of top k retrievals from train set (shape: n x max(ks)).
            ks (list): List of k values to compute metrics at.

        Returns:
            dict: Dictionary of retrieval metrics.
        """
        metrics = {'overall': {}}
        for k in ks:
            metrics['overall'][f'top{k}_acc'] = self.acc_at_k(preds, labels, k)
            metrics['overall'][f'mv@{k}_acc'] = self.mv_acc_at_k(preds, labels, k)
            metrics['overall'][f'mAP@{k}'] = self.map_at_k(preds, labels, k)

        # Compute metrics per class
        class_metrics = {}
        for cls in range(self.num_classes):
            cls_indices = (labels == cls)
            cls_labels = labels[cls_indices]
            cls_preds = preds[cls_indices]

            cls_metrics = {'support': len(cls_labels)}
            for k in ks:
                cls_metrics[f'top{k}_acc'] = self.acc_at_k(cls_preds, cls_labels, k)
                cls_metrics[f'mv@{k}_acc'] = self.mv_acc_at_k(cls_preds, cls_labels, k)
                cls_metrics[f'mAP@{k}'] = self.map_at_k(cls_preds, cls_labels, k)
            class_metrics[int(cls)] = cls_metrics

        metrics['per_class'] = class_metrics  # Add per-class metrics to the overall metrics

        if saveto is not None:
            os.makedirs(os.path.dirname(saveto), exist_ok=True)
            with open(saveto, 'w') as f:
                json.dump(metrics, f, indent=4)

        return metrics
    
    @staticmethod
    def acc_at_k(retrievals, y_queries, k):
        """
        Calculate accuracy at k.

        Args:
            retrievals (torch.Tensor): Labels of top k retrievals from train set (shape: n x max(ks)).
            y_queries (torch.Tensor): True labels for queries (shape: n).
            k (int): The 'k' in 'accuracy at k'.

        Returns:
            float: Accuracy at k.
        """
        topk_preds = retrievals[:, :k] # Shape: n x k
        return torch.any(topk_preds == y_queries[:, None], dim=1).float().mean().item() # if any of the topk matches, then it's correct

    @staticmethod
    def mv_acc_at_k(retrievals, y_queries, k):
        """
        Calculate majority vote accuracy at k.

        Args:
            retrievals (torch.Tensor): Labels of top k retrievals from train set (shape: n x max(ks)).
            y_queries (torch.Tensor): True labels for queries (shape: n).
            k (int): The 'k' in 'majority vote accuracy at k'.

        Returns:
            float: Majority vote accuracy at k.
        """
        topk_preds = retrievals[:, :k] # Shape: n x k
        all_uniques = [torch.unique(row, return_counts=True) for row in topk_preds] # Get majority vote for each row
        outcomes = []
        for label, (uniques, counts) in zip(y_queries, all_uniques):
            max_count = torch.max(counts)
            modes = uniques[counts == max_count] # If there are multiple modes, then it's a tie
            outcome = torch.isin(label, modes, assume_unique=True) # Check if label is in modes
            outcomes.append(outcome)
        return torch.tensor(outcomes).float().mean().item()
    
    @staticmethod
    def map_at_k(retrievals, y_queries, k):
        """
        Calculate mean average precision at k.

        Args:
            retrievals (torch.Tensor): Labels of top k retrievals from train set (shape: n x max(ks)).
            y_queries (torch.Tensor): True labels for queries (shape: n).
            k (int): The 'k' in 'mean average precision at k'.

        Returns:
            float: Mean average precision at k.
        """
        average_precisions = []
        for i, query_label in enumerate(y_queries):
            topk_labels = retrievals[i, :k]
            correct_count = 0
            precision_at_k = 0.0
            for j, label in enumerate(topk_labels):
                if label == query_label:
                    correct_count += 1
                    precision_at_k += correct_count / (j + 1)
            if correct_count > 0:
                average_precision = precision_at_k / k
            else:
                average_precision = 0.0
            average_precisions.append(average_precision)
        return np.mean(average_precisions)