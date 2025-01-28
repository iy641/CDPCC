import torch
import torch.nn as nn
import numpy as np

class NTXentLoss(nn.Module):
    """
    NT-Xent Loss for contrastive learning.
    
    Args:
        batch_size (int): Number of samples in each batch.
        device: Device to run the loss calculation on (e.g., 'cpu' or 'cuda').
        temperature: Temperature scaling parameter (tau) for contrastive learning.
        use_cosine_similarity (boolean): Whether to use cosine similarity (if True) or dot product as the similarity metric (if False).
    """

    def __init__(self, batch_size, temperature, use_cosine_similarity, device):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity) # Similarity function (cosine similarity or dot product)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        """
        Returns the similarity function (cosine similarity or dot product).
        """
        if use_cosine_similarity:
            self._cosine_similarity = nn.CosineSimilarity(dim=-1)
            return self._cosine_similarity
        else:
            return self._dot_similarity

    def _get_correlated_mask(self):
        """
        Creates a mask to exclude diagonal and positive pairs in the similarity matrix.
        """
        diag = np.eye(2 * self.batch_size)  # Diagonal elements
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)  # Lower diagonal for positive pairs
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)  # Upper diagonal for positive pairs
        mask = torch.from_numpy((diag + l1 + l2))  # Combine diagonals
        mask = (1 - mask).type(torch.bool)  
        return mask

    @staticmethod
    def _dot_similarity(x, y):
        """
        Computes dot product similarity between two sets of vectors.
        """
        return torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)

    @staticmethod
    def _cosine_similarity(x, y):
        """
        Computes cosine similarity between two sets of vectors.
        """
        return nn.CosineSimilarity(dim=-1)(x.unsqueeze(1), y.unsqueeze(0))

    def forward(self, zi, zj):
        """
        Compute NT-Xent loss given two augmented views of a batch.

        Args:
            zi: Embeddings of the first augmented view (shape: [batch_size, embedding_dim]).
            zj: Embeddings of the second augmented view (shape: [batch_size, embedding_dim]).

        Returns:
            Loss: Scalar NT-Xent loss.
        """
        # Move inputs to the specified device
        zi, zj = zi.to(self.device), zj.to(self.device)

        # Concatenate embeddings from both augmented views
        representations = torch.cat([zj, zi], dim=0)

        # Compute similarity matrix for all pairs
        similarity_matrix = self.similarity_function(representations, representations)

        # Extract positive pair similarities (l_pos and r_pos are diagonals)
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        # Mask out positive samples to extract negatives
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        # Combine positives and negatives into logits
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature  # Apply temperature scaling

        # Create labels for cross-entropy loss (positives are always the first column)
        labels = torch.zeros(2 * self.batch_size).long().to(self.device)

        # Compute cross-entropy loss
        loss = self.criterion(logits, labels)

        # Normalize loss by the batch size
        return loss / (2 * self.batch_size)
