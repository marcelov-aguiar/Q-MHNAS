import numpy as np
import torch


class MetricTracker:
    """
    A class for tracking evaluation metrics during training and validation,
    supporting both classification and regression tasks.

    Parameters
    ----------
    task : str
        The type of task. Must be either 'multi-class' for classification or 'regression' for regression problems.

    Attributes
    ----------
    task : str
        The task type.
    total : float
        Accumulated number of samples or accumulated loss value.
    correct : int
        Accumulated number of correct predictions (only for classification).
    count : int
        Number of batches (used for averaging loss in regression).
    """

    def __init__(self, task: str):
        self.task = task
        self.total = 0
        self.correct = 0
        self.count = 0

    def update(self, y_logits: torch.Tensor, labels: torch.Tensor):
        """
        Update the internal state based on model predictions and ground truth labels.

        For classification tasks ('multi-class'), accumulates the number of correct predictions and total samples.
        For regression tasks ('regression'), accumulates the root mean squared error (RMSE) over batches.

        Parameters
        ----------
        y_logits : torch.Tensor
            Model predictions. Shape: (batch_size, num_classes) for classification,
            (batch_size, 1) or (batch_size,) for regression.

        labels : torch.Tensor
            Ground truth labels. Shape should match y_logits accordingly.
        """
        if self.task == 'multi-class' or self.task == 'classification':
            _, predicted = y_logits.max(1)
            self.total += labels.size(0)
            self.correct += predicted.eq(labels).sum().item()
        
        elif self.task == 'regression':
            # reduction='sum' soma o erro de TODOS os elementos (Batch * Horizon)
            sse = torch.nn.functional.mse_loss(y_logits, labels, reduction='sum').item()
            self.total += sse
            
            # CORREÇÃO: Usar .numel() para contar (Batch * Horizon)
            # Antes estava labels.size(0) que contava apenas o Batch
            self.count += labels.numel() 
            
        else:
            raise ValueError(f"Unknown task type: {self.task}")

    def result(self) -> float:
        """
        Compute the final metric based on accumulated values.

        Returns
        -------
        float
            The average metric:
            - Accuracy (%) for classification.
            - Root Mean Squared Error (RMSE) for regression.
        """
        if self.task == 'multi-class' or self.task == 'classification':
            return 100.0 * self.correct / self.total if self.total > 0 else 0.0
        elif self.task == 'regression':
            mse = self.total / self.count if self.count > 0 else 0.0
            return float(np.sqrt(mse))