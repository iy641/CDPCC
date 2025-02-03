import torch
import numpy as np
import random
import io
import sys
import logging
from typing import Optional, Tuple


class EarlyStopper:

    def __init__(self, patience = 1, min_delta = 0.0, restore_best_weights = True, monitor = "loss") :
        """
         Early stopping class to stop training when no improvement is seen after a given patience.

        Args:
            patience (int): Number of epochs with no improvement before stopping training.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            restore_best_weights (bool): Restore model weights from the epoch with the best metric value.
            monitor (str): Metric to monitor ('loss' or 'accuracy').
        """

        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.counter = 0
        self.best_metric = float("inf") if monitor == "loss" else float("-inf")
        self.best_model_params = None
        self.best_classifier_params = None

    def should_stop_training(self, metric_value: float, model = None, classifier = None):
        
        """
        Determine whether to stop training based on the metric value.

        Args:
            metric_value (float): Current value of the monitored metric.
            model (Optional[torch.nn.Module]): The model (i.e., the encoder) being trained. 
            classifier (Optional[torch.nn.Module]): The classifier being trained.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """

        if ( (self.monitor == "loss" and metric_value < self.best_metric)
            or (self.monitor == "accuracy" and metric_value > self.best_metric)
        ): # if improvement
            self.best_metric = metric_value
            self.counter = 0
            if self.restore_best_weights:
                if model is not None:
                    self.best_model_params = self.model_parameters(model)
                if classifier is not None:
                    self.best_classifier_params = self.model_parameters(classifier)
        elif (
            (self.monitor == "loss" and metric_value > self.best_metric + self.min_delta)
            or (self.monitor == "accuracy" and metric_value < self.best_metric - self.min_delta)
        ): # if no improvement

            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    @staticmethod
    def model_parameters (model: torch.nn.Module) :
        """
        Save the model parameters.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            dict: Model state dictionary.
        """

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        return torch.load(buffer)

    def get_best_params(self) :

        """
        Retrieve the best model and classifier parameters.

        Returns:
            Tuple[Optional[dict], Optional[dict]]: Best model parameters and classifier parameters if available, None otherwise.
        """
        
        return self.best_model_params, self.best_classifier_params


def set_requires_grad(model , requires_grad = False):
    
    """
    Set the requires_grad attribute of all parameters in the model.

    Args:
        model (torch.nn.Module): The model.
        requires_grad (bool): Whether to require gradients for the model parameters (If yes = parameters will be updated; no = parameters are frozen).
    """

    for param in model.parameters():
        param.requires_grad = requires_grad


def fix_randomness(seed):
    """
    Fix randomness in torch, numpy, and random for reproducibility.

    Args:
        seed (int): The seed value to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(logger_name, level= logging.DEBUG) : 

    """
    Create and return a custom logger with the given name and level.

    Args:
        logger_name (str): The name of the logger.
        level (int): Logging level (default: logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    log_format = logging.Formatter("%(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(logger_name, mode="a")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger

