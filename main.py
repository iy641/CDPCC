
import os
import torch
import numpy as np
from datetime import datetime
import argparse

# Import custom modules
from dataloader import data_generator
from engine import Trainer
from models import CDPCC_Model, LinearClassifier
from utils import fix_randomness, get_logger



# Record the start time of the experiment
start_time = datetime.now()

parser = argparse.ArgumentParser()
home_dir = os.getcwd()

# Define command-line arguments
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Run description')
parser.add_argument('--seed', default=42, type=int,
                    help='Seed value for reproducibility')
parser.add_argument('--training_mode', default='linear', type=str,
                    help='Training mode: linear, supervised, random_init')
parser.add_argument('--selected_dataset', default='CSTH', type=str,
                    help='Dataset of choice: CSTH, Arc_Loss, FD_A')
parser.add_argument('--logs_save_dir', default='../experiments_logs', type=str,
                    help='Directory to save logs and experiment outputs')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')

# Parse the command-line arguments
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'#################### We are using {device} now ####################')

experiment_description = args.experiment_description
dataset = args.selected_dataset
method = 'CDPCC'
training_mode = args.training_mode
run_description = args.run_description

# Create the logs save directory if it doesn't exist
logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

# Import the configuration based on the selected dataset.
# This expects a file like 'config_files/CSTH_Configs.py' with a Config class.
exec(f'from config_files.{dataset}_Configs import Config as Configs')
configs = Configs()

# Fix random seeds for reproducibility
SEED = args.seed
fix_randomness(SEED)

# Create a directory to store logs for this experiment.
experiment_log_dir = os.path.join(logs_save_dir, experiment_description, f"{training_mode}_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# Initialize logging with a unique log file name based on the current date and time.
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = get_logger(log_file_name)
logger.debug("=" * 60)
logger.debug(f'Dataset: {dataset}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug (f'Seed:   {SEED}')
logger.debug("=" * 60)

# Load datasets from the corresponding directory.
data_path = os.path.join(".", "datasets", dataset)
train_loader, val_loader, test_loader = data_generator(configs, sourcedata_path=data_path)
logger.debug("#################### Data loaded ... ####################")

# Initialize the model and classifier, and move them to the selected device.
model = CDPCC_Model(configs).to(device)
classifier = LinearClassifier(configs).to(device)

# Execute different training pipelines based on the training mode.
if training_mode == 'random_init':
    # Mode: random_init
    # Only the classifier is trained while the CDPCC model (encoder) remains with random weights.
    classifier_optim = torch.optim.Adam(classifier.parameters(),
                                        lr=configs.lr,
                                        betas=(configs.beta1, configs.beta2))
    # Train the classifier.
    _, best_classifier_params = Trainer(
        configs,
        model=model,
        classifier=classifier,
        classifier_optim=classifier_optim,
        train_dl=train_loader,
        valid_dl=val_loader,
        mode='linear',
        early_stopping=True,
        device=device
    )
    classifier.load_state_dict(best_classifier_params)

elif training_mode == 'linear':
    # Mode: linear
    # First, pre-train the CDPCC model (encoder) using contrastive loss.
    model_optim = torch.optim.Adam(model.parameters(),
                                   lr=configs.lr,
                                   betas=(configs.beta1, configs.beta2))
    classifier_optim = torch.optim.Adam(classifier.parameters(),
                                        lr=configs.lr,
                                        betas=(configs.beta1, configs.beta2))

    best_model_params, _ = Trainer(
        configs,
        model=model,
        model_optim=model_optim,
        train_dl=train_loader,
        valid_dl=val_loader,
        mode='pre_train',
        early_stopping=True,
        device=device
    )
    model.load_state_dict(best_model_params)

    # Next, train the classifier on top of the frozen model.
    _, best_classifier_params = Trainer(
        configs,
        model=model,
        classifier=classifier,
        classifier_optim=classifier_optim,
        train_dl=train_loader,
        valid_dl=val_loader,
        mode='linear',
        early_stopping=True,
        device=device
    )
    classifier.load_state_dict(best_classifier_params)

elif training_mode == 'supervised':
    # Mode: supervised
    # Jointly train both the CDPCC model (encoder) and the classifier.
    model_optim = torch.optim.Adam(model.parameters(),
                                   lr=configs.lr,
                                   betas=(configs.beta1, configs.beta2))
    classifier_optim = torch.optim.Adam(classifier.parameters(),
                                        lr=configs.lr,
                                        betas=(configs.beta1, configs.beta2))

    best_model_params, best_classifier_params = Trainer(
        configs,
        model=model,
        model_optim=model_optim,
        classifier=classifier,
        classifier_optim=classifier_optim,
        train_dl=train_loader,
        valid_dl=val_loader,
        mode='supervised',
        early_stopping=True,
        device=device
    )
    model.load_state_dict(best_model_params)
    classifier.load_state_dict(best_classifier_params)

# Evaluate the final trained model and classifier on the test set.
_, _, _, _, performance = Trainer(
    configs,
    model=model,
    classifier=classifier,
    test_dl=test_loader,
    mode='test',
    device=device
)

logger.debug(f"Testing Performance:\n"
             f"  - Accuracy (%)  = {performance [0] * 100:.4f}\n"
             f"  - Precision (%) = {performance [1] * 100:.4f}\n"
             f"  - Recall (%)    = {performance [2] * 100:.4f}\n"
             f"  - F1 (%)        = {performance [3] * 100:.4f}")

logger.debug("=" * 60)

logger.debug(f"Training time is : {datetime.now()-start_time}")
