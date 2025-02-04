# Time Series Representation Learning Via Cross-Domain Predictive and Contextual Contrasting: Application to Fault Detection

This repository is the official implementation of [PAPER](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5085741). 

_by: Ibrahim Yousef, Sirish L. Shah, and R. Bhushan Gopaluni_

## Overview

<p align="center">
<img src="images/CDPCC_Figure.png" width="800" class="center">
</p>


Data-driven methods for fault detection increasingly rely on large historical datasets, yet annotations are costly and time-consuming. As a result, learning approaches that minimize the need for extensive labeling, such as self-supervised learning (SSL), are becoming more popular. Contrastive learning, a subset of SSL, has shown promise in fields like computer vision and natural language processing, yet its application in fault detection is not fully explored. In this paper, we propose **Cross-Domain Predictive and Contextual Contrasting (CDPCC)**, a novel contrastive learning framework designed to extract informative latent representations from time series data. CDPCC is specifically designed to capture the cross-domain dynamics between time and frequency features of time series signals. The framework first splits the time series into non-overlapping frames, applying FFT to each frame to create its spectral view. The *cross-domain predictive contrasting module* then learns correlations and dynamic patterns between the time and frequency domains. In addition, we propose a *cross-domain contextual contrasting module* to capture discriminative features. We evaluate CDPCC on fault detection tasks using both simulated and industrial benchmark datasets. Experimental results demonstrate that a linear classifier trained on the features learned by CDPCC performs comparably to fully supervised models. Moreover, CDPCC proves highly efficient in few-labelled and transfer learning scenarios—achieving superior performance with only 50\% of labelled data compared to fully supervised training on the entire labelled dataset.


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

CDPCC has been implemented using Python = 3.11

## Datasets

We consider three publicly available benchmark datasets: 

- [CSTH](https://zenodo.org/records/10093059) (dataset can be found under /datasets/CSTH or [HERE](https://doi.org/10.5683/SP3/8FXNGM)
- [Arc Loss](https://www.sciencedirect.com/science/article/pii/S0959152423001105) (dataset can be downloaded from [Dataverse](https://doi.org/10.5683/SP3/NREPZM))
- FD

### Configurations

The hyper-parameters of the CDPCC model used for each dataset can be found in ```config_files/dataset_name_Configs.py```

### Data Preparation

To prepare the datasets, create a subfolder for each dataset inside  ```datasets/```. Each dataset folder should contain three files: ```train.pt```, ```val.pt```, and ```test.pt```. The data in these files should be stored in a dictionary format. For train.pt, it should contain the training data under the key ```["samples"]``` and the corresponding labels under the key ```["labels"]```. Similarly, ```val.pt``` should contain the validation data and labels, and ```test.pt``` should contain the test data and labels, following the same dictionary structure with ```["samples"]``` for the data and ```["labels"]``` for the labels.

## Running the code

This project provides several training modes, which can be selected via command-line arguments. You can choose from the following training modes:

- **Random Initialization** (`random_init`): Only the classifier is trained, while the CDPCC model (encoder) remains with random weights.
- **Linear** (`linear`): Pre-train the CDPCC model using contrastive loss and then train the classifier on top of the frozen model.
- **Supervised** (`supervised`): Jointly train both the CDPCC model (encoder) and the classifier.

In addition, you can set the experiment description, run description, and seed value for reproducibility.

To run the code with the desired settings, use the following command:

```python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode random_init --selected_dataset CSTH```

Where:
- `experiment_description` is a name for your experiment (e.g., `exp1`).
- `run_description` is a name for a specific run (e.g., `run_1`).
- `seed` sets the random seed value for reproducibility.
- `training_mode` specifies the training mode (choose from `random_init`, `linear`, or `supervised`).
- `selected_dataset` refers to the dataset you want to use.

Make sure the dataset folder is located in the `data` directory and the dataset name matches exactly what is specified in the `--selected_dataset` argument (e.g., `CSTH` for the `CSTH` dataset).


## Citation

```
@article{Ibrahim_Yousef
  title   = {Time Series Representation Learning Via Cross-Domain Predictive and Contextual
            Contrasting: Application to Fault Detection},
  author  = {Yousef, Ibrahim and Shah, Sirish L. and Gopaluni, R. Bhushan},
  journal = {Engineeing Application of Artificial Intelligence},
  year    = {2025}
}
```

## Contact

Please feel free to ask any questions you might have about the code and/or the work to <iy641@mail.ubc.ca>

## Licence

The CDPCC code is released under the MIT license.
