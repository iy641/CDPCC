# Time Series Contrastive Learning via Cross-Domain Predictive and Contextual Contrasting: Application to Fault Detection

#### Authors: Ibrahim Yousef, Sirish L. Shah, and R. Bhushan Gopaluni

#### CDPCC Paper: [Link]

## Overview 

This repository provides the code and datasets for the proposed CDPCC model, along with baseline models, as presented in the manuscript *"Time Series Contrastive Learning via Cross-Domain Predictive and Contextual Contrasting: Application to Fault Detection"*.

We introduce Cross-Domain Predictive and Contextual Contrasting (CDPCC), a novel contrastive learning framework that integrates temporal and spectral information to capture rich time-frequency features from time series data. CDPCC consists of two key components: **i)** cross-domain predictive contrasting, which predicts future embeddings across time and frequency domains, and **ii)** cross-domain contextual contrasting, which aligns time- and frequency-based representations in a shared latent space.

<p align="center">
    <img src="images/CDPCC_Figure.png" width="1000" align="center">
</p>

Our CDPCC model has six components: time- and frequency-domain encoders ($g_E^T$ and $g_E^F$), autoregressive models ($g_{AR}^T$ and $g_{AR}^F$), and non-linear projection heads ($g_P^T$ and $g_P^F$). First, the input time series $x$ is split into $K$ non-overlapping frames $\mathbf{s^T}$, each transformed into its spectral view $\mathbf{s^F}$ via FFT. In the cross-domain predictive contrasting module, time-based and frequency-based representations are produced ($\mathbf{h^T} = g_E^T(\mathbf{s^T})$ and $\mathbf{h^F} = g_E^F(\mathbf{s^F})$). The autoregressive models summarize the dynamics of the first $k_{\text{past}}$ frames (here, $k_{\text{past}} = 3$) to generate context vectors, which are then used to predict future embeddings in the other domain. The cross-domain contextual contrasting module further aligns these context vectors to learn discriminative feature representations. 
