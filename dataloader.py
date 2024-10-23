import os
import torch
from typing import Dict, Tuple
from torch.utils.data import Dataset, DataLoader
import torch.fft as fft
from einops import rearrange

# Function to perform frequency transformation on input data
def frequency_transform(x_in: torch.Tensor, frame_size: int) -> torch.Tensor:
    """
    Perform FFT on the input frames and extract only the important frequency components.

    Args:
        x_in (torch.Tensor): Input tensor of shape (samples, input variables, number of frames, frame size).
        frame_size (int): Size of each frame.

    Returns:
        torch.Tensor: Tensor with important frequency components.
    """
    n_important_freq = frame_size // 2 + 1
    # Perform FFT on the last dimension and take the magnitude of the important frequencies
    X_patches_F = fft.fft(x_in, dim=3)[:, :, :, :n_important_freq].abs()
    return X_patches_F


# Custom PyTorch dataset class for loading and transforming data
class Load_Dataset(Dataset):
    def __init__(self, dataset: Dict[str, torch.Tensor], frame_size: int):
        """
        Custom PyTorch dataset for loading and patching time series data.

        Args:
            dataset (Dict[str, torch.Tensor]): A dictionary containing 'samples' and 'labels' tensors.
            frame_size (int): Size of each frame.

        Returns:
            None
        """
        super().__init__()
        
        self.X_data = dataset["samples"]
        self.y_data = dataset["labels"]
        self.seq_length = dataset["samples"].shape[-1]  # Sequence length of the input data

        # Ensure that X_data has at least 3 dimensions
        if len(self.X_data.shape) < 3:
            self.X_data = self.X_data.unsqueeze(2)

        # Ensure channels are in the second dimension (N, V, T)
        if self.X_data.shape.index(min(self.X_data.shape)) != 1:
            self.X_data = self.X_data.permute(0, 2, 1)

        # Validate frame size
        if frame_size > self.seq_length:
            raise ValueError("Frame size cannot be greater than the sequence length.")
        
        if not isinstance(frame_size, int) or frame_size <= 0:
            raise ValueError("Frame size must be a positive integer.")

        self.frame_size = frame_size

        # Handle padding if sequence length is not divisible by frame size
        if self.seq_length % self.frame_size == 0:
            self.num_frames = self.seq_length // frame_size
        else:
            # Add padding to fit the patch size
            remainder = self.seq_length % self.frame_size
            padding_size = self.frame_size - remainder
            pad_left = padding_size // 2
            pad_right = padding_size - pad_left
            self.X_data = torch.nn.functional.pad(self.X_data, (pad_left, pad_right), mode='constant', value=0)
            self.num_frames = (self.seq_length // frame_size) + 1

        # Rearrange the data into frames (num of samples, num of variables, num of frames, frame size)
        self.X_frames_T = rearrange(self.X_data, 'N v (p n) -> N v p n', p=self.num_frames, n=self.frame_size)
        
        # Perform frequency transformation on the frames
        self.X_frames_F = frequency_transform(self.X_frames_T, self.frame_size)

        # Dataset length
        self.len = self.X_data.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample and its corresponding label.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Time-domain frames, frequency-domain frames, and label.
        """
        return self.X_frames_T[index], self.X_frames_F[index], self.y_data[index]

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.len


# Function to generate data loaders for training, validation, and testing datasets
def data_generator(config: object, sourcedata_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing datasets.

    Args:
        config (object): config
        sourcedata_path (str): Path to the directory containing preprocessed datasets.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Data loaders for training, validation, and testing datasets.
    """
    # Load preprocessed datasets from file
    train_dataset = torch.load(os.path.join(sourcedata_path, "train.pt"))
    val_dataset = torch.load(os.path.join(sourcedata_path, "val.pt"))
    test_dataset = torch.load(os.path.join(sourcedata_path, "test.pt"))

    # Create dataset instances
    train_dataset = Load_Dataset(train_dataset, frame_size= config.frame_size)
    val_dataset = Load_Dataset(val_dataset, frame_size= config.frame_size)
    test_dataset = Load_Dataset(test_dataset, frame_size= config.frame_size)

    # Create data loaders for each dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size= config.batch_size,
                              shuffle= T, drop_last=config.drop_last, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size= config.batch_size,
                            shuffle=False, drop_last=config.drop_last, num_workers= 0)
    test_loader = DataLoader(dataset=test_dataset, batch_size= config.batch_size,
                             shuffle=False, drop_last= False, num_workers= 0)

    return train_loader, val_loader, test_loader
