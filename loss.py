import torch
import numpy as np

# NT-Xent (Normalized Temperature-scaled Cross Entropy) loss function used in contrastive learning
class NTXentLoss(torch.nn.Module):
    """
    NT-Xent Loss for contrastive learning, commonly used in self-supervised learning approaches.
    
    Args:
    device: Device to run the loss calculation on (e.g., 'cpu' or 'cuda').
    batch_size: Number of samples in each batch.
    temperature: Temperature scaling parameter (tau) for contrastive learning.
    use_cosine_similarity: Whether to use cosine similarity or dot product as the similarity metric.
    """

    def __init__(self, batch_size, temperature, use_cosine_similarity, device):
        """
        Initializes the NT-Xent Loss with given parameters.
        
        Args:
        device: Device to run the loss calculation on.
        batch_size: Number of samples in each batch.
        temperature: Temperature parameter (tau) to scale the logits.
        use_cosine_similarity: Boolean to choose between cosine similarity or dot product for the similarity function.
        """
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)

        # Mask to ignore positive samples in the similarity matrix
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)

        # Choose similarity function (cosine similarity or dot product)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)

        # CrossEntropy loss with sum reduction
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        """
        Returns the similarity function (cosine similarity or dot product).
        
        Args:
        use_cosine_similarity: Boolean indicating whether to use cosine similarity.
        
        Returns:
        Function: The similarity function to use.
        """
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        """
        Creates a mask to ignore positive samples in the similarity matrix.
        
        Returns:
        Tensor: A mask to zero out positive samples in the similarity matrix.
        """
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        """
        Computes dot product similarity between two sets of vectors.
        
        Args:
        x: Tensor of shape (N, C).
        y: Tensor of shape (2N, C).
        
        Returns:
        Tensor: Dot product similarity of shape (N, 2N).
        """
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        """
        Computes cosine similarity between two sets of vectors.
        
        Args:
        x: Tensor of shape (N, C).
        y: Tensor of shape (2N, C).
        
        Returns:
        Tensor: Cosine similarity of shape (N, 2N).
        """
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        """
        Forward pass to compute the NT-Xent loss.
        
        Args:
        zis: Tensor of shape (N, C) representing the first set of augmented samples.
        zjs: Tensor of shape (N, C) representing the second set of augmented samples.
        
        Returns:
        Tensor: Computed NT-Xent loss.
        """
        # Concatenate the representations of both views (zis and zjs)
        representations = torch.cat([zjs, zis], dim=0)

        # Compute similarity matrix for all pairs
        similarity_matrix = self.similarity_function(representations, representations)

        # Extract positive sample similarities (l_pos and r_pos represent positive pairs)
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        # Mask out positive samples from the similarity matrix to obtain negatives
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        # Concatenate positive and negative similarities
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature  # Apply temperature scaling

        # Labels: positive samples are always the first in the concatenated logits
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()

        # Compute the cross-entropy loss
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)  # Normalize by batch size
