import torch
import numpy as np
import random
import io
import sys
import logging

# Early stopping class to stop training when no improvement is seen after a given patience
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, restore_best_weights=True, monitor='loss'):
        """
        Initialize EarlyStopper object.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best metric value.
            monitor (str): Metric to monitor ('loss' or 'accuracy').
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = float('inf') if monitor == 'loss' else float('-inf')
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.best_model_params = None
        self.best_classifier_params = None

    # Check whether to stop training based on the current metric
    def should_stop_training(self, metric_value, model=None, classifier=None):
        """
        Determine whether to stop training based on the metric value.

        Args:
            metric_value (float): Current value of the monitored metric.
            model: The model being trained.
            classifier: The classifier being trained.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        # If an improvement in the monitored metric is observed
        if (self.monitor == 'loss' and metric_value < self.best_metric) or (self.monitor == 'accuracy' and metric_value > self.best_metric):
            self.best_metric = metric_value
            self.counter = 0
            # Optionally save the best model and classifier parameters
            if self.restore_best_weights:
                if model is not None:
                    self.best_model_params = self.serialize_model(model)
                if classifier is not None:
                    self.best_classifier_params = self.serialize_model(classifier)
        # If no improvement is seen and patience is exceeded
        elif (self.monitor == 'loss' and metric_value > self.best_metric + self.min_delta) or (self.monitor == 'accuracy' and metric_value < self.best_metric - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    # Serialize the model's state dictionary for later restoration
    def serialize_model(self, model):
        """
        Serialize the model's state dictionary.

        Args:
            model (torch.nn.Module): The model to be serialized.

        Returns:
            dict: Serialized model state dictionary.
        """
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        return torch.load(buffer)

    # Retrieve the best model and classifier parameters
    def get_best_params(self):
        """
        Get the best model parameters.

        Returns:
            tuple: Best model parameters and classifier parameters if available, None otherwise.
        """
        return self.best_model_params, self.best_classifier_params

# Utility function to set requires_grad attribute for model parameters
def set_requires_grad(model, requires_grad=False):
    """
    Set the requires_grad attribute of all parameters in the model.

    Args:
        model (torch.nn.Module): The model whose parameters' requires_grad will be modified.
        requires_grad (bool): Whether to require gradients for the model parameters.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad

# Fix random seeds for reproducibility
def fix_randomness(seed):
    """
    Fixes randomness in torch, numpy, and random for reproducibility.

    Args:
        seed (int): The seed value to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger
