import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import NTXentLoss
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, f1_score, recall_score
from utils import EarlyStopper, set_requires_grad


def Trainer (config, model, classifier=None, model_optim=None, classifier_optim=None, train_dl=None, valid_dl=None,
                 test_dl=None, mode='pre_train', early_stopping=True, device='cpu'):
    """
    Function to train and evaluate a model and classifier.

    Args:
        config (object): Configuration object containing hyperparameters and settings.
        model (torch.nn.Module): The model.
        classifier (torch.nn.Module, optional): The classifier.
        model_optim (torch.optim.Optimizer, optional): Optimizer for the model.
        classifier_optim (torch.optim.Optimizer, optional): Optimizer for the classifier.
        train_dl (torch.utils.data.DataLoader, optional): DataLoader providing batches of training data.
        valid_dl (torch.utils.data.DataLoader, optional): DataLoader providing batches of validation data.
        test_dl (torch.utils.data.DataLoader, optional): DataLoader providing batches of testing data.
        mode (str, optional): Training mode ('pretrain', 'linear', 'supervised', or 'test'). Defaults to 'pretrain'.
        early_stopping (bool, optional): Whether to use early stopping. Defaults to True.
        device (str, optional): The device to move tensors to. Defaults to 'cpu'.

    Returns:
        tuple: Test loss, test accuracy, logits, and predictions if mode is 'test', or best model and classifier parameters.
    """
    valid_modes = ['pre_train', 'linear', 'supervised', 'test']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Mode must be one of {valid_modes}.")

    best_model_params = None
    best_classifier_params = None
    
    if mode == 'test':
        print('########### CDPCC Testing Results ###########')
        test_loss, test_acc, test_logits, test_predictions, test_labels, test_performance = evaluate(
            model, classifier, test_dl, mode, config, device)
        return test_loss, test_acc, test_logits, test_predictions, test_labels, test_performance
    
    if early_stopping:
        monitor_metric = 'accuracy' if mode != 'pre_train' else 'loss'
        early_stopper = EarlyStopper(patience=config.patience, monitor=monitor_metric)

    print('############### CDPCC ###############')

    for epoch in range(1, config.num_epoch + 1):
        if mode == 'pre_train':
            train_loss, train_acc = pretrain(model, model_optim, train_dl, config, device)
            valid_loss, valid_acc = evaluate(model, classifier, valid_dl, mode, config, device)
        elif mode == 'linear':
            train_loss, train_acc = linear_train(model, classifier, classifier_optim, train_dl, config, device)
            valid_loss, valid_acc = evaluate(model, classifier, valid_dl, mode, config, device)
        elif mode == 'supervised':
            train_loss, train_acc = supervised_train(model, classifier, model_optim, classifier_optim, train_dl, config, device)
            valid_loss, valid_acc = evaluate(model, classifier, valid_dl, mode, config, device)

        print(f'\nEpoch: {epoch}\n'
              f'Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}\n'
              f'Train Accuracy: {train_acc:.4f} | Valid Accuracy: {valid_acc:.7f}')

        if early_stopping:
            if early_stopper.should_stop_training(metric_value=valid_loss if mode == 'pre_train' else valid_acc, 
                                                  model=model, classifier=classifier):
                print('############# Early Stopping Triggered! #############')
                break

            if early_stopper.get_best_params()[0] is not None:
                best_model_params, best_classifier_params = early_stopper.get_best_params()

    print('#################### CDPCC (pre-)training is over ####################')        

    if early_stopping:
        return best_model_params, best_classifier_params
    else:
        return model.state_dict().copy(), (classifier.state_dict().copy() if classifier else None)


def pretrain(model, model_optim, dataloader, config, device):
    """
    Pretrain a model using contrastive learning.

    Args:
        model (torch.nn.Module): The model to be pretrained.
        model_optim (torch.optim.Optimizer): Optimizer for model parameters.
        dataloader (torch.utils.data.DataLoader): Training data loader.
        config (object): Configuration containing hyperparameters.
        device (str): Device to move tensors to.

    Returns:
        tuple: Training loss and accuracy.
    """
    losses = []
    model.train()

    for (X_T, X_F, y) in dataloader:
        X_T, X_F, y = X_T.float().to(device), X_F.float().to(device), y.long().to(device)
        model_optim.zero_grad()

        _, _, _, _, z_T, z_F, nce_TtoF, nce_FtoT = model(X_T, X_F)

        NTXent_loss = NTXentLoss(config.batch_size, config.temperature_coeff, config.cosine_similarity, device)
        contextual_loss = NTXent_loss(z_T, z_F)

        loss = config.lambda_1 * (nce_TtoF + nce_FtoT) + config.lambda_2 * contextual_loss
        losses.append(loss.item())

        loss.backward()
        model_optim.step()

    total_loss = torch.tensor(losses).mean()
    total_acc = 0  # Not computed during pretraining
    return total_loss, total_acc


def linear_train(model, classifier, classifier_optim, dataloader, config, device):
    """
    Train a linear classifier on top of a pre-trained encoder.

    Args:
        model (torch.nn.Module): Pre-trained encoder model.
        classifier (torch.nn.Module): Linear classifier model.
        classifier_optim (torch.optim.Optimizer): Optimizer for classifier parameters.
        dataloader (torch.utils.data.DataLoader): Training data loader.
        config (object): Configuration containing hyperparameters.
        device (str): Device to move tensors to.

    Returns:
        tuple: Training loss and accuracy.
    """
    losses, accuracies = []
    set_requires_grad(model, False)  # Freeze encoder parameters
    classifier.train()

    for (X_T, X_F, y) in dataloader:
        X_T, X_F, y = X_T.float().to(device), X_F.float().to(device), y.long().to(device)
        classifier_optim.zero_grad()

        h_T, h_F, _, _, _, _, _, _ = model(X_T, X_F)
        h_conc = torch.cat((h_T.reshape(config.batch_size, -1), h_F.reshape(config.batch_size, -1)), dim=1)

        logits, predictions, loss, accuracy = classifier(h_conc, y)

        losses.append(loss.item())
        accuracies.append(accuracy.item())

        loss.backward()
        classifier_optim.step()

    total_loss = torch.tensor(losses).mean()
    total_acc = torch.tensor(accuracies).mean()
    return total_loss, total_acc


def supervised_train(model, classifier, model_optim, classifier_optim, dataloader, config, device):
    """
    Train both a pre-trained encoder and linear classifier in supervised learning.

    Args:
        model (torch.nn.Module): Pre-trained encoder model.
        classifier (torch.nn.Module): Linear classifier model.
        model_optim (torch.optim.Optimizer): Optimizer for encoder parameters.
        classifier_optim (torch.optim.Optimizer): Optimizer for classifier parameters.
        dataloader (torch.utils.data.DataLoader): Training data loader.
        config (object): Configuration containing hyperparameters.
        device (str): Device to move tensors to.

    Returns:
        tuple: Training loss and accuracy.
    """
    losses, accuracies = []
    model.train()
    classifier.train()

    for (X_T, X_F, y) in dataloader:
        X_T, X_F, y = X_T.float().to(device), X_F.float().to(device), y.long().to(device)
        model_optim.zero_grad()
        classifier_optim.zero_grad()

        h_T, h_F, _, _, _, _, _, _ = model(X_T, X_F)
        h_conc = torch.cat((h_T.reshape(config.batch_size, -1), h_F.reshape(config.batch_size, -1)), dim=1)

        logits, predictions, loss, accuracy = classifier(h_conc, y)

        losses.append(loss.item())
        accuracies.append(accuracy.item())

        loss.backward()
        model_optim.step()
        classifier_optim.step()

    total_loss = torch.tensor(losses).mean()
    total_acc = torch.tensor(accuracies).mean()
    return total_loss, total_acc


def evaluate(model, classifier, dataloader, mode, config, device):
    """
    Evaluate a model and classifier on a dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        classifier (torch.nn.Module): The classifier to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of data for evaluation.
        mode (str): Evaluation mode ('pretrain', 'linear', 'supervised', or 'test').
        config (object): Configuration object containing hyperparameters and settings.
        device (str): The device to move tensors to.

    Returns:
        tuple: Loss, accuracy, logits, and predictions.
    """
    losses = []
    accuracies = []
    logits = np.empty((0, config.num_classes))
    outputs = []
    labels_np_all = []
    predictions_np_all = []

    model.eval()
    if mode != 'pre_train':
        classifier.eval()

    with torch.no_grad():
        for (X_T, X_F, y) in dataloader:
            X_T, X_F, y = X_T.float().to(device), X_F.float().to(device), y.long().to(device)

            if mode == 'pre_train':
                _, _, _, _, z_T, z_F, nce_TtoF, nce_FtoT = model(X_T, X_F)

                NTXent_loss = NTXentLoss(config.batch_size, config.temperature_coeff, config.cosine_similarity, device)
                contextual_loss = NTXent_loss(z_T, z_F)
                loss = config.lambda_1 * (nce_TtoF + nce_FtoT) + config.lambda_2 * contextual_loss
                losses.append(loss.item())

            else:
                h_T, h_F, _, _, _, _, _, _ = model(X_T, X_F)
                h_conc = torch.cat((h_T.reshape(config.batch_size, -1), h_F.reshape(config.batch_size, -1)), dim=1)

                logit, prediction, loss, accuracy = classifier(h_conc, y)
                losses.append(loss.item())
                accuracies.append(accuracy.item())

                logits = np.append(logits, logit.cpu().numpy(), axis=0)
                labels_np_all.extend(y.detach().cpu().numpy())
                predictions_np_all.extend(logit.detach().cpu().numpy())
                outputs.extend(np.argmax(logit.detach().cpu().numpy(), axis=1))

    total_loss = torch.tensor(losses).mean()

    if mode == 'pre_train':
        return total_loss, 0

    elif mode == 'linear' or mode == 'supervised':
        return total_loss, torch.tensor(accuracies).mean()

    else:
        labels_np_all = np.array(labels_np_all)
        outputs = np.array(outputs)
        predictions_np_all = np.array(predictions_np_all)

        onehot_y_all = np.eye(config.num_classes)[labels_np_all]

        total_acc = torch.tensor(accuracies).mean().item()
        precision = precision_score(labels_np_all, outputs, average='macro')
        recall = recall_score(labels_np_all, outputs, average='macro')
        F1 = f1_score(labels_np_all, outputs, average='macro')
        total_auc = roc_auc_score(onehot_y_all, predictions_np_all, average="macro", multi_class="ovr")
        total_prc = average_precision_score(onehot_y_all, predictions_np_all, average="macro")

        performance = [total_acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100]
        print(f'CDPCC Testing: Accuracy = {total_acc*100:.4f}| Precision = {precision*100:.4f} | '
              f'Recall = {recall*100:.4f} | F1 = {F1*100:.4f} | AUROC = {total_auc*100:.4f} | AUPRC = {total_prc*100:.4f}')

        return total_loss, total_acc, predictions_np_all, outputs, labels_np_all, performance