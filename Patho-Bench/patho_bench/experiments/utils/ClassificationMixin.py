import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, precision_recall_curve, f1_score, balanced_accuracy_score, precision_score, recall_score, cohen_kappa_score, classification_report
from sklearn.preprocessing import label_binarize
sns.set_style("white")

"""
Contains visualization methods for classification tasks
"""

class ClassificationMixin:
    def auc_roc(self, y_true, preds, num_classes, saveto=None):
        """
        Generates multi-class ROC curves and computes AUC for each class ('one-vs-rest')

        Args:
            y_true (Array[int]): 1D ground truth labels for each sample
            preds (Array[float]): N-dimensional predictions for each sample, where N is the number of classes
            num_classes (int): Number of classes
            saveto (str, optional): Path to save figure
        """
        y_true_bin = self.binarize_labels(y_true, num_classes)

        fig, ax = plt.subplots(figsize=(10, 10))

        # Compute auc scores for each class
        auc_scores = roc_auc_score(y_true_bin, preds, average=None, multi_class='ovr')

        # Plot ROC curve for each class
        for j in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, j], preds[:, j])
            ax.plot(fpr, tpr, label=f"Class {j} (OVR AUC={auc_scores[j]:.2f})", color=sns.color_palette("Set1", n_colors=num_classes)[j])

        # ROC curve styling
        ax.plot([0, 1], [0, 1], linestyle="--", color="k", linewidth=0.8)
        ax.set_xlabel("False Positive Rate", fontsize=14)
        ax.set_ylabel("True Positive Rate", fontsize=14)
        ax.set_title(f"ROC Curve ({num_classes} classes)", fontsize=16, pad=20)
        ax.legend(loc="lower right", frameon=False, fontsize=12)
        
        fig.tight_layout()
        if saveto is not None:
            os.makedirs(os.path.dirname(saveto), exist_ok=True)
            fig.savefig(saveto)
        plt.close()

    def precision_recall(self, y_true, preds, num_classes, saveto=None):
        """
        Generates precision-recall curves and computes AUC for each class in multi-class classification.

        Args:
            y_true (Array[int]): 1D ground truth labels for each sample
            preds (Array[float]): N-dimensional predictions for each sample, where N is the number of classes
            num_classes (int): Number of classes
            saveto (str, optional): Path to save figure
        """
        y_true_bin = self.binarize_labels(y_true, num_classes)

        fig_pr, ax_pr = plt.subplots(figsize=(10, 10))

        # Plot PR curve for each class
        for j in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, j], preds[:, j])
            auc_pr = auc(recall, precision)
            ax_pr.plot(recall, precision, label=f"Class {j} (OVR PR={auc_pr:.2f})", color=sns.color_palette("Set1", n_colors=num_classes)[j])

            # Compute no-skill line
            no_skill = sum(y_true_bin[:, j]) / len(y_true_bin[:, j])
            ax_pr.plot([0, 1], [no_skill, no_skill], linestyle="--", color=sns.color_palette("Set1", n_colors=num_classes)[j], linewidth=0.8,)

        # PR curve styling
        ax_pr.set_xlabel("Recall", fontsize=14)
        ax_pr.set_ylabel("Precision", fontsize=14)
        ax_pr.set_title(f"Precision-Recall Curve ({num_classes} classes)", fontsize=16, pad=20)
        ax_pr.legend(frameon=False, fontsize=12)
        ax_pr.set_xlim([0, 1])
        ax_pr.set_ylim([0, 1])

        fig_pr.tight_layout()
        if saveto is not None:
            os.makedirs(os.path.dirname(saveto), exist_ok=True)
            fig_pr.savefig(saveto)
        plt.close()

    def confusion_matrix(self, y_true, preds, num_classes, threshold=None, saveto=None):
        """
        Generates a confusion matrix for multi-class classification.

        Args:
            y_true (Array[int]): 1D ground truth labels for each sample
            preds (Array[float]): N-dimensional predictions for each sample, where N is the number of classes
            num_classes (int): Number of classes
            threshold (float, optional): Custom threshold for classification. If None, use highest score.
            saveto (str, optional): Path to save figure
        """
        y_pred = self.get_predicted_labels(preds, threshold)

        fig_cm, ax_cm = plt.subplots(figsize=(10, 10))

        matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("True label")
        ax_cm.set_xticklabels(list(range(num_classes)))
        ax_cm.set_yticklabels(list(range(num_classes)))
        ax_cm.set_title(f"Confusion Matrix with T={threshold:.2f}" if threshold is not None else "Confusion Matrix")
        ax_cm.set_aspect("equal")
        ax_cm.collections[0].colorbar.remove()

        fig_cm.tight_layout()
        if saveto is not None:
            os.makedirs(os.path.dirname(saveto), exist_ok=True)
            fig_cm.savefig(saveto)
        plt.close()

    def classification_metrics(self, y_true, preds, num_classes, threshold=None, saveto=None):
        """
        Computes precision, recall, accuracy, and other metrics for multi-class classification.

        Args:
            y_true (Array[int]): 1D ground truth labels for each sample
            preds (Array[float]): N-dimensional predictions for each sample, where N is the number of classes
            num_classes (int): Number of classes
            threshold (float, optional): Threshold for classification. If None, will use the class with the highest score.
            saveto (str, optional): Path to save JSON file with scores
        """
        y_pred = self.get_predicted_labels(preds, threshold)

        scores = {"threshold": threshold}

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0, labels=list(range(num_classes)))

        # Compute overall metrics
        if num_classes == 2:
            preds = preds[:, 1]  # Only use column for positive class
        
        try:
            macro_ovr_auc = roc_auc_score(y_true, preds, average="macro", multi_class="ovr", labels = list(range(num_classes)))
            macro_ovo_auc = roc_auc_score(y_true, preds, average="macro", multi_class="ovo", labels = list(range(num_classes)))
        except Exception as e:
            print(f"WARNING: Could not compute AUC scores because {e}. Will set to None.")
            macro_ovr_auc = None
            macro_ovo_auc = None

        scores["overall"] = {
            "macro-ovr-auc": macro_ovr_auc,
            "macro-ovo-auc": macro_ovo_auc,
            "macro-precision": report["macro avg"]["precision"],
            "macro-recall": report["macro avg"]["recall"],
            "macro-f1": report["macro avg"]["f1-score"],
            "weighted-precision": report["weighted avg"]["precision"],
            "weighted-recall": report["weighted avg"]["recall"],
            "weighted-f1": report["weighted avg"]["f1-score"],
            "acc": np.mean(np.array(y_true) == np.array(y_pred)),
            "bacc": balanced_accuracy_score(y_true, y_pred),
            "weighted_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
            "support": report["macro avg"]["support"]
        }

        # Save scores to JSON
        if saveto is not None:
            os.makedirs(os.path.dirname(saveto), exist_ok=True)
            with open(saveto, "w") as f:
                json.dump(scores, f, indent=4)

        return scores

    def binarize_labels(self, y_true, num_classes):
        """
        Converts integer labels to one-hot encoded format

        Args:
            y_true (Array[int]): 1D ground truth labels for each sample
            num_classes (int): Number of classes
            
        Returns:
            Array[int]: 2D one-hot encoded labels
        """
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        # Add column if num_classes == 2, because label_binarize only returns one column (for positive class) for binary classification
        return np.hstack((1 - y_true_bin, y_true_bin)) if num_classes == 2 else y_true_bin
    
    def get_predicted_labels(self, preds, threshold=None):
        """
        Gets predicted labels from predictions

        Args:
            preds (Array[float]): 2D predicted probabilities (n_samples x n_classes)
            threshold (float, optional): Minimum score for a class to be considered a positive prediction. If None, will use the class with the highest score.

        Returns:
            Array[int]: 1D predicted labels (n_samples)
        """
        if threshold is not None:
            return np.array([np.argmax(row) if max(row) > threshold else -1 for row in preds])
        else:
            return np.argmax(preds, axis=1)