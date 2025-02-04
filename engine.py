import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import NTXentLoss
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import EarlyStopper, set_requires_grad


def Trainer(config, model, classifier=None, model_optim=None, classifier_optim=None,
            train_dl=None, valid_dl=None, test_dl=None, mode='pre_train',
            early_stopping=True, device='cpu'):
  
    """
    Train and evaluate the model and optional classifier using different training protocols.

    Modes:
        - 'pre_train': Contrastive pretraining. (input: the model only)
        - 'linear': Linear classifier training with frozen encoder. (inputs: the model and classifier)
        - 'supervised': Joint supervised training of encoder and classifier. (inputs: the model and classifier)
        - 'test': Evaluation only on the test set. (inputs: the model and classifier)

    Args:
        config (object): Configuration with hyperparameters and settings.
        model (torch.nn.Module): The encoder (i.e., the CDPCC model).
        classifier (torch.nn.Module, optional): The classifier model (linear layer), if applicable.
        model_optim (torch.optim.Optimizer, optional): Optimizer for the model (needed for 'pre_train' and 'supervised' modes).
        classifier_optim (torch.optim.Optimizer, optional): Optimizer for the classifier (needed for 'linear' and 'supervised' modes).
        train_dl (torch.utils.data.DataLoader, optional): Training data loader.
        valid_dl (torch.utils.data.DataLoader, optional): Validation data loader.
        test_dl (torch.utils.data.DataLoader, optional): Testing data loader.
        mode (str, optional): Training/evaluation mode. Defaults to 'pre_train'.
        early_stopping (bool, optional): Whether to use early stopping. Defaults to True.
        device (str, optional): Device identifier (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        If mode == 'test':
            tuple: (test_loss, test_accuracy, test_logits, test_predictions, test_labels, test_performance)

        Else:
            tuple: (best_model_params, best_classifier_params) if early_stopping is True;
                   otherwise, the final state_dict(s).
    """

    list_modes = ['pre_train', 'linear', 'supervised', 'test']
    if mode not in list_modes:
        raise ValueError(f"Invalid mode '{mode}'. Mode must be one of {list_modes}.")

    best_model_params = None
    best_classifier_params = None

    # Test mode: Only evaluate on the test set.
    if mode == 'test':
        print('#################### CDPCC Testing Results ####################')
        test_loss, test_acc, test_predictions, test_labels, test_performance = evaluate(
            config, model, classifier, test_dl, mode, device)
        return test_loss, test_acc, test_predictions, test_labels, test_performance

    # Initialize early stopper if needed.
    if early_stopping:
        # For pretraining, monitor loss; otherwise, monitor accuracy.
        monitor_metric = 'accuracy' if mode != 'pre_train' else 'loss'
        early_stopper = EarlyStopper(patience=config.patience, monitor=monitor_metric)

    print('#################### CDPCC Training ####################')

    for epoch in range(1, config.num_epoch + 1):
        # Depending on the mode, perform the appropriate training step.
        if mode == 'pre_train':
            train_loss, train_acc = pretrain(config, model, model_optim, train_dl, device)
            valid_loss, valid_acc = evaluate(config, model, classifier, valid_dl, mode, device)
        elif mode == 'linear':
            train_loss, train_acc = linear_train(config, model, classifier, classifier_optim, train_dl, device)
            valid_loss, valid_acc = evaluate(config, model, classifier, valid_dl, mode, device)
        elif mode == 'supervised':
            train_loss, train_acc = supervised_train(config, model, classifier, model_optim, classifier_optim, train_dl, device)
            valid_loss, valid_acc = evaluate(config, model, classifier, valid_dl, mode, device)

        # Log epoch details.
        print(f'\nEpoch: {epoch}\n'
              f'Train Loss:     {train_loss:.4f} | Valid Loss:     {valid_loss:.4f}\n'
              f'Train Accuracy: {train_acc:.4f} | Valid Accuracy: {valid_acc:.4f}')

        # Check for early stopping.
        if early_stopping:
            # For pretraining use loss, otherwise use accuracy as the metric.
            current_metric = valid_loss if mode == 'pre_train' else valid_acc
            if early_stopper.should_stop_training(metric_value=current_metric,
                                                    model=model, classifier=classifier):
              
                print('#################### Early Stopping Triggered! ####################')
                break

            # Save best model and classifier parameters.
            best_model_params, best_classifier_params = early_stopper.get_best_params()

    print('#################### CDPCC (Pre-)Training is over ####################')

    if early_stopping:
        return best_model_params, best_classifier_params
    else:
        # Return the final state dictionaries if not using early stopping.
        return model.state_dict().copy(), (classifier.state_dict().copy() if classifier else None)


def pretrain(config, model, model_optim, dataloader, device):
    """
    Pre-train the model using contrastive loss.

    Args:
        config (object): Configuration with hyperparameters.
        model (torch.nn.Module): The CDPCC model to pre-train.
        model_optim (torch.optim.Optimizer): Optimizer for model parameters.
        dataloader (torch.utils.data.DataLoader): Data loader for training data.
        device (str): Device identifier.

    Returns:
        tuple: (training loss, training accuracy)
    """

    losses = []
    model.train()  # Set model to training mode.

    for (X_T, X_F, y) in dataloader:
        # Move batch data to the target device and cast appropriately.
        X_T = X_T.float().to(device)
        X_F = X_F.float().to(device)
        y = y.long().to(device)

        model_optim.zero_grad()  # Reset gradients.

        # Forward pass: obtain embeddings and cross-domain predictive contrasting losses.
        _, _, _, _, z_T, z_F, nce_TtoF, nce_FtoT = model(X_T, X_F)

        # Compute the cross-domain contextual contrasting loss.
        ntxent_loss = NTXentLoss(config.batch_size, config.temperature_coeff, config.cosine_similarity, device)
        contextual_loss = ntxent_loss(z_T, z_F)

        # Combine losses with respective coefficients.
        loss = config.lambda_1 * (nce_TtoF + nce_FtoT) + config.lambda_2 * contextual_loss
        losses.append(loss.item())

        loss.backward()  # Backpropagation.
        model_optim.step()  # Update model parameters.

    # Compute the average loss over all batches.
    total_loss = torch.tensor(losses).mean()
    total_acc = 0  # Accuracy is not computed during pretraining. 
    return total_loss, total_acc


def linear_train(config, model, classifier, classifier_optim, dataloader, device):
    """
    Train a linear classifier on top of a frozen pre-trained encoder.

    Args:
        config (object): Configuration with hyperparameters.
        model (torch.nn.Module): Pre-trained CDPCC model.
        classifier (torch.nn.Module): Linear classifier model.
        classifier_optim (torch.optim.Optimizer): Optimizer for classifier parameters.
        dataloader (torch.utils.data.DataLoader): Data loader for training data.
        device (str): Device identifier.

    Returns:
        tuple: (training loss, training accuracy)
    """
    losses = []
    accuracies = []

    # Freeze CDPCC parameters.
    set_requires_grad(model, False)
    classifier.train()  # Set classifier to training mode.

    for (X_T, X_F, y) in dataloader:
        # Move data to device.
        X_T = X_T.float().to(device)
        X_F = X_F.float().to(device)
        y = y.long().to(device)

        classifier_optim.zero_grad()

        # Get encoded represnetations.
        h_T, h_F, _, _, _, _, _, _ = model(X_T, X_F)
        # concatenate time- and frequency-domain encoded representations into one vector.
        h_conc = torch.cat((h_T.reshape(config.batch_size, -1),
                            h_F.reshape(config.batch_size, -1)), dim=1)

        # Forward pass through the classifier.
        logits, predictions, loss, accuracy = classifier(h_conc, y)
        losses.append(loss.item())
        accuracies.append(accuracy.item())

        loss.backward()  # Backpropagation for classifier.
        classifier_optim.step()

    total_loss = torch.tensor(losses).mean()
    total_acc = torch.tensor(accuracies).mean()
    return total_loss, total_acc


def supervised_train(config, model, classifier, model_optim, classifier_optim, dataloader, device):
    """
    Jointly train the CDPCC model and classifier in a supervised manner (no-pretraining).

    Args:
        config (object): Configuration with hyperparameters.
        model (torch.nn.Module): The CDPCC model.
        classifier (torch.nn.Module): Classifier model.
        model_optim (torch.optim.Optimizer): Optimizer for CDPCC parameters.
        classifier_optim (torch.optim.Optimizer): Optimizer for classifier parameters.
        dataloader (torch.utils.data.DataLoader): Data loader for training data.
        device (str): Device identifier.

    Returns:
        tuple: (training loss, training accuracy)
    """
    losses = []
    accuracies = []
    model.train()       # Set CDPCC model to training mode.
    classifier.train()  # Set classifier to training mode.

    for (X_T, X_F, y) in dataloader:
     
        X_T = X_T.float().to(device)
        X_F = X_F.float().to(device)
        y = y.long().to(device)

        model_optim.zero_grad()
        classifier_optim.zero_grad()

        # Forward pass through encoder.
        h_T, h_F, _, _, _, _, _, _ = model(X_T, X_F)
        # Flatten and concatenate features.
        h_conc = torch.cat((h_T.reshape(config.batch_size, -1),
                            h_F.reshape(config.batch_size, -1)), dim=1)

        # Forward pass h_conc through classifier.
        logits, predictions, loss, accuracy = classifier(h_conc, y)
        losses.append(loss.item())
        accuracies.append(accuracy.item())

        loss.backward()  # Backpropagation through both networks (CDPCC and linear classifier).
        model_optim.step()
        classifier_optim.step()

    total_loss = torch.tensor(losses).mean()
    total_acc = torch.tensor(accuracies).mean()
    return total_loss, total_acc


def evaluate(config, model, classifier, dataloader, mode, device):
    """
    Evaluate the model (and classifier if applicable) on a given dataset.

    Args:
        config (object): Configuration with hyperparameters.
        model (torch.nn.Module): The CDPCC model.
        classifier (torch.nn.Module): The classifier (if mode != 'pre-traine').
        dataloader (torch.utils.data.DataLoader): Data loader for evaluation data.
        mode (str): Evaluation mode ('pre_train', 'linear', 'supervised', or 'test').
        device (str): Device identifier.

    Returns:
        For mode ='pre_train':
            tuple: (loss,  0)
        For model ='linear' or 'supervised':
            tuple: (loss,  accuracy)
        For mode ='test':
            tuple: (loss, accuracy, y_logit, y_pred, y_acual, performance metrics)
    """
    losses = []
    accuracies = []
    # Initialize arrays/lists to store outputs and labels.
    y_preds = []
    y_actuals = []

    model.eval()  # Set CDPCC to evaluation mode.
    if mode != 'pre_train':
        classifier.eval()  # Set classifier to evaluation mode if mode !='pre_train'.

    with torch.no_grad():
        for (X_T, X_F, y) in dataloader:

            X_T = X_T.float().to(device)
            X_F = X_F.float().to(device)
            y = y.long().to(device)

            if mode == 'pre_train':
                # For pretraining, only compute the contrastive loss.
                _, _, _, _, z_T, z_F, nce_TtoF, nce_FtoT = model(X_T, X_F)
                ntxent_loss = NTXentLoss(config.batch_size, config.temperature_coeff, config.cosine_similarity, device)
                contextual_loss = ntxent_loss(z_T, z_F)
                loss = config.lambda_1 * (nce_TtoF + nce_FtoT) + config.lambda_2 * contextual_loss
                losses.append(loss.item())
            else:
                # For supervised/linear modes, perform a full forward pass.
                h_T, h_F, _, _, _, _, _, _ = model(X_T, X_F)
                print (h_T.shape)
                h_conc = torch.cat((h_T.reshape(config.batch_size, -1),
                                    h_F.reshape(config.batch_size, -1)), dim=1)

                _, y_pred, loss, accuracy = classifier (h_conc, y)
                losses.append(loss.item())
                accuracies.append(accuracy.item())

                # Accumulate actual labels and predictions for performance metrics.
                y_actuals.extend(y.cpu().numpy())
                y_preds.extend(y_pred.cpu().numpy())

    total_loss = torch.tensor(losses).mean()

    if mode == 'pre_train':
        return total_loss, 0

    elif mode in ['linear', 'supervised']:
        # Return average loss and accuracy.
        return total_loss, torch.tensor(accuracies).mean()

    else:  # Test mode: compute additional performance metrics.
        y_actuals = np.array(y_actuals)
        y_preds = np.array(y_preds)

        total_acc = torch.tensor(accuracies).mean().item()
        precision = precision_score(y_actuals, y_preds, average='binary')
        recall = recall_score(y_actuals, y_preds, average='binary')
        f1 = f1_score(y_actuals, y_preds, average='binary')

        performance = [total_acc * 100, precision * 100, recall * 100,f1 * 100]

        print(f'CDPCC Testing: Accuracy (%) = {total_acc * 100:.4f} | '
              f'Precision (%) = {precision * 100:.4f} | Recall (%) = {recall * 100:.4f} | '
              f'F1 (%) = {f1 * 100:.4f}')

        return total_loss, total_acc, y_preds, y_actuals, performance
