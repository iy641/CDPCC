import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np
from typing import Union, List, Tuple

# Encoder block for processing temporal or spectral frames
class FrameEncoder(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 kernel_size: int, stride: int, num_hidden_layers: int, 
                 dropout: float, padding: int):
        """
        Segment encoder that applies convolutional layers to input data.

        Args:
            in_channels (int): Number of input variables.
            hidden_channels (int): Number of hidden channels (number of feature maps).
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size for the convolutions.
            stride (int): Stride for the convolutions.
            num_hidden_layers (int): Number of hidden convolutional layers. We consider only one in the study.
            dropout (float): Dropout rate for regularization.
            padding (int): Padding for the convolutions. We consider no padding (i.e., 'valid'). 
        """
        super().__init__()
        # Input convolutional block
        self.conv_block_in = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Hidden convolutional layers
        self.conv_blocks_hidden = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_hidden_layers)
        ])
        # Output convolutional block
        self.conv_block_out = nn.Sequential(
            nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FrameEncoder.

        Args:
            x (torch.Tensor): Input frame in tensor format.

        Returns:
            torch.Tensor: Encoded representation of the input frame.
        """
        x = self.conv_block_in(x)
        for hidden_layer in self.conv_blocks_hidden:
            x = hidden_layer(x)
        x = self.conv_block_out(x)
        return x


# Encoder for time-domain frames. Note: Both g_E^T and g_E^T follow the same model architecutre. 
class BaseEncoder_T(nn.Module):

    def __init__(self, config):
        """
        Time-domain encoder that uses the FrameEncoder to process time-domain frames.

        Args:
            config: Configuration object containing model parameters.
        """

        super().__init__()
        self.encoder = FrameEncoder(
            in_channels=config.input_variables, hidden_channels=config.num_kernels_hidden,
            out_channels=config.num_kernels_out, kernel_size=config.kernel_size,
            stride=config.stride, num_hidden_layers=config.hidden_layers,
            dropout=config.conv_dropout, padding=config.conv_padding
        )

        # Dynamically compute the input size for the linear projection layer
        dummy_input = torch.randn(1, config.input_variables, config.frame_size)
        with torch.no_grad():
            in_size = int(self.encoder(dummy_input).shape[-1] * config.num_kernels_out)

        self.linear = nn.Linear(in_features=in_size, out_features=config.features_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the time-domain encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_variables, num_frames, frame_size).

        Returns:
            torch.Tensor: Encoded output.
        """
        batch_size, num_variables, num_frames, frame_size = x.size()
        x = x.permute(0, 2, 1, 3)  # Reshape input for frame encoders
        x = [self.encoder(x[:, i, :, :]) for i in range(num_frames)]
        x = torch.stack(x, dim=1).flatten(start_dim=2)  # Flatten the frames
        x = self.linear(x)
        return x


# Encoder for frequency-domain frames
class BaseEncoder_F(nn.Module):
    def __init__(self, config):
        """
        Frequency-domain encoder that uses the FrameEncoder to process spectral frames.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        self.encoder = FrameEncoder(
            in_channels=config.input_variables, hidden_channels=config.num_kernels_hidden,
            out_channels=config.num_kernels_out, kernel_size=config.kernel_size,
            stride=config.stride, num_hidden_layers=config.hidden_layers,
            dropout=config.conv_dropout, padding=config.conv_padding
        )

        # Dynamically compute the input size for the linear layer
        dummy_input = torch.randn(1, config.input_variables, config.frame_size // 2 + 1)
        with torch.no_grad():
            in_size = int(self.encoder(dummy_input).shape[-1] * config.num_kernels_out)

        self.linear = nn.Linear(in_features=in_size, out_features=config.features_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the frequency-domain encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_variables, num_frames, frame_size).

        Returns:
            torch.Tensor: Encoded output.
        """
        batch_size, num_variables, num_frames, frame_size = x.size()
        x = x.permute(0, 2, 1, 3)  # Reshape input for segment encoders
        x = [self.encoder(x[:, i, :, :]) for i in range(num_frames)]
        x = torch.stack(x, dim=1).flatten(start_dim=2)  # Flatten the frames
        x = self.linear(x)
        return x


# LSTM module for autoregressive modeling
class LSTMContext(nn.Module):
    def __init__(self, config):
        """
        LSTM module for autoregressive modeling.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()

        self.lstm = nn.LSTM(config.features_length, config.context_dim, 
                            num_layers=config.lstm_layers, batch_first=True, 
                            dropout=config.lstm_dropout)

        self.batch_size = config.batch_size
        self.context_dim = config.context_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM context module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Context vector.
        """

        lstm_out, _ = self.lstm(x)
        context_vector = lstm_out[:, -1, :].reshape(self.batch_size, self.context_dim)
        return context_vector


# Simple projection head used in contrastive learning. We use two non-linear layers to project context vectors 
# into a lower-dimensional space

class ProjectionHead(nn.Module):
    def __init__(self, config):
        """
        Projection head used in contrastive learning.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(in_features=config.context_dim, out_features=config.context_dim // 2),
            nn.BatchNorm1d(config.context_dim // 2),
            nn.ReLU(),
            nn.Linear(in_features=config.context_dim // 2, out_features=config.context_dim // 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projection head.

        Args:
            x (torch.Tensor): Input context (c).

        Returns:
            torch.Tensor: Projected output (z).
        """
        return self.projection_head(x)


# Main CDPCC model
class CDPCC_Model (nn.Module):
    def __init__(self, config):
        """
        CDPCC model with time and frequency encoders.

        Args:
            config: Configuration object containing model parameters.
        """

        super().__init__()
        self.timestep = config.n_future_timesteps
        self.Wk_TtoF = nn.ModuleList([nn.Linear(config.context_dim, config.features_length) for _ in range(self.timestep)])
        self.Wk_FtoT = nn.ModuleList([nn.Linear(config.context_dim, config.features_length) for _ in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=-1)

        self.base_encoder_T = BaseEncoder_T(config)
        self.base_encoder_F = BaseEncoder_F(config)

        self.AR_T = LSTMContext(config)
        self.AR_F = LSTMContext(config)

        self.projection_head_T = ProjectionHead(config)
        self.projection_head_F = ProjectionHead(config)

        self.device = config.device

    def forward(self, x_T: torch.Tensor, x_F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the CDPCC model.

        Args:
            x_T (torch.Tensor): Time-domain input tensor.
            x_F (torch.Tensor): Frequency-domain input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Encoded representations and loss values.
        """
        x_T, x_F = x_T.to(self.device), x_F.to(self.device)

        h_T = self.base_encoder_T(x_T).squeeze()
        h_F = self.base_encoder_F(x_F).squeeze()

        batch_size, num_frames, h_dim = h_T.shape

        future_embeddings_T = h_T[:, -self.timestep:, :].transpose(0, 1)
        future_embeddings_F = h_F[:, -self.timestep:, :].transpose(0, 1)

        past_embeddings_T = h_T[:, :-self.timestep, :]
        past_embeddings_F = h_F[:, :-self.timestep, :]

        c_T = self.AR_T(past_embeddings_T)
        c_F = self.AR_F(past_embeddings_F)

        nce_TtoF, nce_FtoT = 0, 0

        pred_future_emb_T = torch.empty((self.timestep, batch_size, h_dim), device=self.device).float()
        pred_future_emb_F = torch.empty((self.timestep, batch_size, h_dim), device=self.device).float()

        for i in np.arange(0, self.timestep):
            linear = self.Wk_TtoF[i]
            pred_future_emb_F[i] = linear(c_T)
        for i in np.arange(0, self.timestep):
            loss_TtoF = torch.mm(future_embeddings_F[i], pred_future_emb_F[i].transpose(0, 1))
            nce_TtoF += torch.sum(torch.diag(self.lsoftmax(loss_TtoF)))
        nce_TtoF /= -1. * batch_size * self.timestep

        for i in np.arange(0, self.timestep):
            linear = self.Wk_FtoT[i]
            pred_future_emb_T[i] = linear(c_F)
        for i in np.arange(0, self.timestep):
            loss_FtoT = torch.mm(future_embeddings_T[i], pred_future_emb_T[i].transpose(0, 1))
            nce_FtoT += torch.sum(torch.diag(self.lsoftmax(loss_FtoT)))
        nce_FtoT /= -1. * batch_size * self.timestep

        return h_T, h_F, c_T, c_F, self.projection_head_T(c_T), self.projection_head_F(c_F), nce_TtoF, nce_FtoT


# Linear classifier used for downstream tasks
class LinearClassifier (nn.Module):
    def __init__(self, config):
        """
        Linear classifier for downstream tasks in CDPCC model.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

        self.input_size = int(config.features_length * config.num_frames * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape = self.input_size),
            nn.Linear(in_features=self.input_size, out_features=config.num_classes)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the linear classifier.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Logits, predictions, loss, and accuracy.
        """
        logits = self.classifier(x)
        predictions = logits.argmax(dim=1)
        loss = self.criterion(logits, y)
        accuracy = y.eq(logits.detach().argmax(dim=1)).float().mean()

        return logits, predictions, loss, accuracy

