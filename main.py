
import torch
import os
import numpy as np
from datetime import datetime
import argparse
from dataloader import data_generator
from trainer import Trainer
from models import CDPCC_Model, LinearClassifier
from utils import fix_randomness, _logger


start_time = datetime.now()

parser = argparse.ArgumentParser()

home_dir = os.getcwd()

parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')

parser.add_argument('--run_description', default='run1', type=str,
                    help='Run Description')

parser.add_argument('--seed', default= 42, type=int,
                    help='seed value')

parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: linear, supervised, random_init')

parser.add_argument('--selected_dataset', default='CSTH', type=str,
                    help='Dataset of choice: CSTH, Arc_Loss, FD_A, FD_B')

parser.add_argument('--logs_save_dir', default='../experiments_logs', type=str,
                    help='saving directory')

# parser.add_argument('--device', default='cuda', type=str,
#                     help='cpu or cuda')

parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')

args = parser.parse_args()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('We are using %s now.' %device)


experiment_description = args.experiment_description
dataset = args.selected_dataset
method = 'CDPCC'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


exec(f'from config_files.{dataset}_Configs import Config as Configs')
configs = Configs()



# # ##### fix random seeds for reproducibility ########
SEED = args.seed
fix_randomness(SEED)

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, training_mode + f"_seed_{SEED}")
# 'experiments_logs/Exp1/run1/linear_seed_0'
os.makedirs(experiment_log_dir, exist_ok=True)




# loop through domains
counter = 0
src_counter = 0



# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
# 'experiments_logs/Exp1/run1/train_linear_seed_0/logs_14_04_2022_15_13_12.log'
logger = _logger(log_file_name)
logger.debug("=" * 30)
logger.debug(f'Dataset: {dataset}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 30)


# Load datasets
data_path = f"./datasets/{dataset}"
train_loader, val_loader, test_loader = data_generator(configs, sourcedata_path = data_path)
logger.debug("Data loaded ...")

# Load Model
model = CDPCC_Model (configs).to(device)
classifier = LinearClassifier (configs).to(device)


if training_mode == 'random_init': 
  
    classifier_optim = torch.optim.Adam(classifier.parameters(),
                                        lr=configs.lr,
                                        betas=(configs.beta1, configs.beta2)
                                        )
    _, best_classifier_params = Trainer(configs,
                                        model=model,
                                        classifier=classifier,
                                        classifier_optim=classifier_optim,
                                        train_dl=train_loader,
                                        valid_dl=val_loader,
                                        mode='linear', early_stopping=True,
                                        device=device)

    classifier.load_state_dict(best_classifier_params)


if training_mode == 'linear': 

    model_optim = torch.optim.Adam(model.parameters(),
                                        lr=configs.lr,
                                        betas=(configs.beta1, configs.beta2)
                                        )
  
    classifier_optim = torch.optim.Adam(classifier.parameters(),
                                        lr=configs.lr,
                                        betas=(configs.beta1, configs.beta2)
                                        )
  
    best_model_params, _ = Trainer(configs,
                                  model=model,
                                  model_optim = model_optim,
                                  train_dl=train_loader,
                                  valid_dl=val_loader,
                                  mode='pre_train', early_stopping=True,
                                  device=device)
  
    model.load_state_dict(best_model_params)

    _, best_classifer_params = Trainer(configs,
                                      model=model,
                                      classifier=classifier,
                                      classifier_optim=classifier_optim,
                                      train_dl=train_loader,
                                      valid_dl=val_loader,
                                      mode='linear', early_stopping=True,
                                      device=device)

    classifier.load_state_dict(best_classifier_params)


if training_mode == 'supervised': 

    model_optim = torch.optim.Adam(model.parameters(),
                                        lr=configs.lr,
                                        betas=(configs.beta1, configs.beta2)
                                        )
  
    classifier_optim = torch.optim.Adam(classifier.parameters(),
                                        lr=configs.lr,
                                        betas=(configs.beta1, configs.beta2)
                                        )
  
    best_model_params, best_classifier_params = Trainer(configs,
                                                        model=model,
                                                        model_optim = model_optim,
                                                        classifier=classifier,
                                                        classifier_optim=classifier_optim,
                                                        train_dl=train_loader,
                                                        valid_dl=val_loader,
                                                        mode='supervised', early_stopping=True,
                                                        device=device)
  
    model.load_state_dict(best_model_params)

    classifier.load_state_dict(best_classifier_params)



_, _, _, _, _, performance = Trainer(configs, model= model,
                                      classifier= classifier,
                                      test_dl=test_loader, mode='test',
                                      device=device)
  
