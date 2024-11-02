# Time Series Contrastive Learning via Cross-Domain Predictive and Contextual Contrasting: Application to Fault Detection

#### Authors: Ibrahim Yousef, Sirish L. Shah, and R. Bhushan Gopaluni

#### CDPCC Paper: [Link]

## Overview 

This repository provides the code and datasets for the proposed CDPCC model, along with baseline models, as presented in the manuscript *"Time Series Contrastive Learning via Cross-Domain Predictive and Contextual Contrasting: Application to Fault Detection"*.

We propose TF-C, a novel pre-training approach for learning generalizable features that can be transferred across different time-series datasets. We evaluate TF-C on eight time series datasets with different sensor measurements and semantic meanings in four real-world application scenarios. The following illustration provides an overview of the idea behind and the broad applicability of our TF-C approach. The idea is shown in **(a)**: given a time series sample, time-based and frequency-based embeddings are made close to each other in a latent time-frequency space. The application scenarios are shown in **(b)**: leveraging TF-C in time series, we can generalize a pre-train models to diverse scenarios such as gesture recognition, fault detection, and seizure analysis.
<!-- Then we fine-tune the models to a small, problem-specific dataset for performing time series classification tasks. -->

<p align="center">
    <img src="images/fig1.png" width="1000" align="center">
</p>
