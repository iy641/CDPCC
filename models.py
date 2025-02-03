import torch
import torch.nn as nn
import numpy as np

class FrameEncoder(nn.Module):

    def __init__(self, in_channels, num_kernels, kernel_size , 
                 stride, num_layers, dropout, padding):
        """
        Encoder block for processing time- or frequency-domain frames using 1D convolutional layers.

        Args:
            in_channels (int): Number of input channels (i.e., variables).
            num_kernels (int): Number of hidden kernels (i.e., number of feature maps).
            kernel_size (int): Size of the 1D convolution kernel.
            stride (int): Stride of the convolution.
            num_layers (int): Number of hidden layers (i.e., number of 1D convolutional layers - 2).
            dropout (float): Dropout rate for regularization.
            padding (int): Padding size for convolution.
        """
        super().__init__()
        
        # Each 1D convolution block contains a 1D convolutional layer followed a non-linear activation function and dropout.
        self.conv_block_in = nn.Sequential(
            nn.Conv1d(in_channels, num_kernels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.conv_blocks_hidden = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_kernels, num_kernels, kernel_size, stride, padding),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.conv_block_out = nn.Sequential(
            nn.Conv1d(num_kernels, num_kernels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input frame of shape (batch_size, num_variables, frame_size).
        Returns:
            torch.Tensor: Encoded frame.
        """

        x = self.conv_block_in(x)
        for hidden_layer in self.conv_blocks_hidden:
            x = hidden_layer(x)   
        return self.conv_block_out(x)


class BaseEncoder_T (nn.Module):

    def __init__(self, config):
        """
        Base encoder (g_E^T) for processing time-domain frames. It consists of a frame encoder and a linear projection layer.

        Args:
            config : Configuration object with model hyperparameters.
        """
        super().__init__()
        
        self.encoder = FrameEncoder(
            in_channels=config.input_variables, num_kernels=config.num_kernels,
            kernel_size=config.kernel_size, stride=config.stride, num_layers=config.num_conv_layers,
            dropout=config.conv_dropout, padding=config.conv_padding
        )
        
        frame_size = config.frame_size
        # Dynamically compute the input size for the linear layer
        dummy_input = torch.randn(1, config.input_variables, frame_size)
        with torch.no_grad():
            in_size = int(self.encoder(dummy_input).shape[-1] * config.num_kernels)
        
        self.linear = nn.Linear(in_size, config.embedding_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input time-domain frames of shape (batch_size, num_variables, num_frames, frame_size).
        Returns:
            torch.Tensor: Encoded time-domain frames (h^T) of shape (batch_size, num_frames, embedding_dim) .
        """

        batch_size, num_variables, num_frames, frame_size = x.size()
        x = x.permute(0, 2, 1, 3)
        x = [self.encoder(x[:, i, :, :]) for i in range(num_frames)]
        x = torch.stack(x, dim=1).flatten(start_dim=2)
        return self.linear(x)

class BaseEncoder_F (nn.Module):

    def __init__(self, config):
        """
        Base encoder (g_E^F) for processing frequency-domain frames. It consists of a frame encoder and a linear projection layer.
        Note: g_E^F has the same architecture as g_E^T. The only difference is that the frequency-domain frame size is time-domain frame size // 2 + 1 
        because when FFT is applied on the time-domain frames, we only extract the important frequency components; the other half is redundant.

        Args:
            config : Configuration object with model hyperparameters.
        """
        super().__init__()
        
        self.encoder = FrameEncoder(
            in_channels=config.input_variables, num_kernels=config.num_kernels,
            kernel_size=config.kernel_size, stride=config.stride, num_layers=config.num_conv_layers,
            dropout=config.conv_dropout, padding=config.conv_padding
        )
        
        frame_size = config.frame_size
        # Dynamically compute the input size for the linear layer
        dummy_input = torch.randn(1, config.input_variables, frame_size)
        with torch.no_grad():
            in_size = int(self.encoder(dummy_input).shape[-1] * config.num_kernels)
        
        self.linear = nn.Linear(in_size, config.embedding_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input frequency-domain frames of shape (batch_size, num_variables, num_frames, frame_size//2 + 1).
        Returns:
            torch.Tensor: Encoded frequency-domain frames (h^F) of shape (batch_size, num_frames, embedding_dim).
        """

        batch_size, num_variables, num_frames, frame_size = x.size()
        x = x.permute(0, 2, 1, 3)
        x = [self.encoder(x[:, i, :, :]) for i in range(num_frames)]
        x = torch.stack(x, dim=1).flatten(start_dim=2)
        return self.linear(x)

class AutoRegressiveModel (nn.Module):
  
    def __init__(self, config):
        """
        LSTM module for autoregressive modeling, g_AR.

        Args:
            config: Configuration object containing model hyperparameters.
        """
        super().__init__()

        self.lstm = nn.LSTM(config.embedding_dim, config.context_dim, 
                            num_layers=config.num_lstm_layers, batch_first=True, 
                            dropout=config.lstm_dropout)
        self.context_dim = config.context_dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_frame up to :k_past, embedding_dim).
        Returns:
            torch.Tensor: Context vector of shape (batch_size, context_dim).
        """
        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :]

class ProjectionHead(nn.Module):
  
    def __init__(self, config):
        """
        Projection head, g_P, (non-linear prokection layers) used in contrastive learning.
        We use two non-linear layers to project context vectors into a lower-dimensional space (1/4 the context dimension).
  
        Args:
            config: Configuration object containing model hyperparameters.
        """

        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(config.context_dim, config.context_dim // 2),
            nn.BatchNorm1d(config.context_dim // 2),
            nn.ReLU(),
            nn.Linear(config.context_dim // 2, config.context_dim // 4)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input context vector of shape (batch_size, context_dim).
        
        Returns:
            torch.Tensor: Projected output, z, of shape (batch_size, context_dim//4).
        """

        return self.projection_head(x)


class CDPCC_Model(nn.Module):

    def __init__(self, config):
        """
        Cross-Domain Predictive and Contextual Contrasting (CDPCC) model.
        The CDPCC model has six components: time- and frequency-domain encoders (g_E^T and g_E^F), 
        two autoregressive models (g_{AR}^T and g_{AR}^F), and two non-linear projection heads (g_P^T and g_P^F)

        Args:
            config: Configuration object containing model hyperparameters.
        """ 

        super().__init__()
        
        self.k_past = config.k_past
        self.k_future = config.num_frame - self.k_past
        self.device = config.device
        
        # Linear transformation layers for predictive contrastive learning
        self.Wk_TtoF = nn.ModuleList([
            nn.Linear(config.context_dim, config.embedding_dim) for _ in range(self.k_future)
        ])
        self.Wk_FtoT = nn.ModuleList([
            nn.Linear(config.context_dim, config.embedding_dim) for _ in range(self.k_future)
        ])
        
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        
        # Encoders for time and frequency domains
        self.base_encoder_T = BaseEncoder_T(config)
        self.base_encoder_F = BaseEncoder_F(config)
        
        # Autoregressive modules. Both AR_T and AR_F have the same architecture, but they do not share parameters
        self.AR_T = AutoRegressiveModel(config)
        self.AR_F = AutoRegressiveModel(config)
        
        # Projection heads. Both projection_head_T and projection_head_F have the same architecture, but they do not share parameters
        self.projection_head_T = ProjectionHead(config)
        self.projection_head_F = ProjectionHead(config)
    
    def forward (self, x_T x_F):
        """
        Forward pass of the CDPCC model.
        
        Args:
            x_T (torch.Tensor): Input time-domain frames.
            x_F (torch.Tensor): Input frequency-domain frames.
        
        Returns:
            Tuple containing encoded representations (h_T and h_F), context vectors (c_T and c_F), and contrastive loss values (nce_TtoF and nce_FtoT).
        """

        x_T, x_F = x_T.to(self.device), x_F.to(self.device)
        
        h_T = self.base_encoder_T(x_T).squeeze()
        h_F = self.base_encoder_F(x_F).squeeze()
        
        batch_size, num_frames, embedding_dim = h_T.shape
        
        # Extract past and future embeddings
        future_embeddings_T = h_T[:, -self.k_future:, :].transpose(0, 1)
        future_embeddings_F = h_F[:, -self.k_future:, :].transpose(0, 1)
        past_embeddings_T = h_T[:, :self.k_past, :]
        past_embeddings_F = h_F[:, :self.k_past, :]
        
        # Compute context representations
        c_T = self.AR_T(past_embeddings_T)
        c_F = self.AR_F(past_embeddings_F)

        # noise Contrastive Estimation (NCE) loss for cross-domain predictive contrasting module.
        #This loss is used in CDPCC to contrast predicted future embeddings using context vectors with actual future embeddings.
        nce_TtoF, nce_FtoT = 0.0, 0.0
        
        pred_future_emb_T = torch.empty((self.k_future, batch_size, embedding_dim), device=self.device).float()
        pred_future_emb_F = torch.empty((self.k_future, batch_size, embedding_dim), device=self.device).float()

        for i in np.arange(0, self.k_future):
            linear = self.Wk_TtoF[i]
            pred_future_emb_F[i] = linear(c_T)
        for i in np.arange(0, self.k_future):
            loss_TtoF = torch.mm(future_embeddings_F[i], pred_future_emb_F[i].transpose(0, 1))
            nce_TtoF += torch.sum(torch.diag(self.lsoftmax(loss_TtoF)))
        nce_TtoF /= -1. * batch_size * self.k_future

        for i in np.arange(0, self.k_future):
            linear = self.Wk_FtoT[i]
            pred_future_emb_T[i] = linear(c_F)
        for i in np.arange(0, self.k_future):
            loss_FtoT = torch.mm(future_embeddings_T[i], pred_future_emb_T[i].transpose(0, 1))
            nce_FtoT += torch.sum(torch.diag(self.lsoftmax(loss_FtoT)))
        nce_FtoT /= -1. * batch_size * self.k_future
      
        return h_T, h_F, c_T, c_F, self.projection_head_T(c_T), self.projection_head_F(c_F), nce_TtoF, nce_FtoT
    

class LinearClassifier(nn.Module):

    def __init__(self, config):
        """
        A linear classifier for downstream tasks in the CDPCC model. 
        The classifier is trained on top of learned representations (h^T and h^F)
        
        Args:
            config: Configuration object containing model hyperparameters.
        """
        super().__init__()
        
        self.criterion = nn.CrossEntropyLoss()
        self.input_size = int(config.embedding_dim * config.num_frames * 2)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.input_size),
            nn.Linear(in_features=self.input_size, out_features=config.num_classes)
        )
    
    def forward(self, x, y):
        """
        Forward pass of the linear classifier.
        
        Args:
            x (torch.Tensor): Input features (i.e., concatenated h^T and h^F vector).
            y (torch.Tensor): Actual labels.
        
        Returns:
            Tuple containing logits, predictions, loss, and accuracy.
        """

        logits = self.classifier(x)
        predictions = logits.argmax(dim=1)
        loss = self.criterion(logits, y)
        accuracy = y.eq(predictions).float().mean()
        
        return logits, predictions, loss, accuracy

