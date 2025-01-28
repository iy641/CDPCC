import os
import torch
from typing import Dict, Tuple
from torch.utils.data import Dataset, DataLoader
import torch.fft as fft
from einops import rearrange

# Function to perform FFT on input frames
def frequency_transform(X_frames_T: torch.Tensor) -> torch.Tensor:

    """
    Perform FFT on the input time-domain frames.

    Args:
        X_frames_T (torch.Tensor): Input time-domain frames of shape (num_samples, num_variables, num_frames K, frame_size d).

    Returns:
        X_frames_F (torch.Tensor): Corresponding frequency-domain frames of shape (num_samples, num_variables, num_frames K, d/2 +1)
    """
    frame_size = X_frames_T.shape[-1]
    n_freq_components = frame_size // 2 + 1
    
    # Perform FFT on each time-domain frame to get the corresponding frequency-domain frame. 
    X_frames_F = fft.fft(X_frames_T, dim=3)[:, :, :, :n_freq_components].abs()
    return X_frames_F


def pad_input (X_data: torch.Tensor, frame_size: int) -> torch.Tensor:
    """
    Pad the input data to make its length divisible by frame_size (d). We add zeros on both ends.

    Args:
        X_data (torch.Tensor): Input data in the shape of (num_samples, num_variables, seq_lenght L).
        frame_size (int): frame size d.

    Returns:
        X_pad (torch.Tensor): Padded input data.
    """
    seq_length = X_data.shape[-1]
    remainder = seq_length % frame_size
    if remainder == 0:  #if the seq_length is divisible --> do nothing
        return X_data

    padding_size = frame_size - remainder
    pad_left = padding_size // 2
    pad_right = padding_size - pad_left
    return torch.nn.functional.pad(X_data, (pad_left, pad_right), mode='constant', value=0) #Zero padding



class Load_Dataset(Dataset):
    def __init__(self, dataset: Dict[str, torch.Tensor], frame_size: int):
        """
        Custom class for initializing, loading, and slicing data

        Args:
            dataset (Dict[str, torch.Tensor]): A dictionary containing 'samples' and 'labels' tensors.
            frame_size (int): Size of each frame (d).

        """

        super().__init__()
        
        self.X_data = dataset["samples"]
        self.y_data = dataset["labels"]
        self.seq_length = dataset["samples"].shape[-1]  # Sequence length of the input data (L)

        # Check that X_data has at least 3 dimensions. X_data should have a shape of (num_samples, num_variables, seq_length L)
        if len(self.X_data.shape) < 3:
            self.X_data = self.X_data.unsqueeze(2) # Add a dimension

        # Check number of variables are in the second dimension
        if self.X_data.shape.index(min(self.X_data.shape)) != 1: 
            self.X_data = self.X_data.permute(0, 2, 1)

        # Validate frame size (should be positive and smaller than the sequence length)
        if frame_size > self.seq_length:
            raise ValueError("Frame size cannot be greater than the sequence length.")
        
        if not isinstance(frame_size, int) or frame_size <= 0:
            raise ValueError("Frame size must be a positive integer.")

        self.frame_size = frame_size

        # Pad the sequence to make it divisible by frame size
        self.X_data = pad_input (self.X_data, frame_size)
        self.num_frames = (self.seq_length // frame_size) + 1

        # Split the data into K non-overlapping frames of the same size : (num_samples, num_variables, num_frames, frame_size)
        self.X_frames_T = rearrange(self.X_data, 'N v (p n) -> N v p n', p=self.num_frames, n=self.frame_size)
        
        # Perform frequency transformation on the frames
        self.X_frames_F = frequency_transform(self.X_frames_T, self.frame_size)

        # Number of samples in the dataset
        self.len = self.X_data.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a specific sample and its corresponding label.

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



def data_generator(config: object, sourcedata_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing datasets.

    Args:
        config (object): config (list of user-specified hyperparameters values)
        sourcedata_path (str): Path to the directory of the dataset.

    Returns:
        train_loader, val_loader, test_loader (Tuple[DataLoader, DataLoader, DataLoader]): Data loaders for training, validation, and testing datasets.
    """

    # Load datasets from from the directory path (assuming the data are stored in .pt files and with the following names. Change if necessary)
    train_dataset = torch.load(os.path.join(sourcedata_path, "train.pt"))
    val_dataset = torch.load(os.path.join(sourcedata_path, "val.pt"))
    test_dataset = torch.load(os.path.join(sourcedata_path, "test.pt"))

    # Create dataset instances
    train_dataset = Load_Dataset(train_dataset, frame_size= config.frame_size)
    val_dataset = Load_Dataset(val_dataset, frame_size= config.frame_size)
    test_dataset = Load_Dataset(test_dataset, frame_size= config.frame_size)

    # Create data loaders for each dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size= config.batch_size,
                              shuffle= True, drop_last= True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size= config.batch_size,
                            shuffle=False, drop_last= True, num_workers= 0)
    test_loader = DataLoader(dataset=test_dataset, batch_size= config.batch_size,
                             shuffle=False, drop_last= False, num_workers= 0)

    return train_loader, val_loader, test_loader
