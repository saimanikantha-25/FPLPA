import torch
import torch.nn as nn
import torch.nn.functional as F

class CPN(nn.Module):
    """
    Convolutional Prototype Network (CPN).
    Takes a 5x5 feature tensor, passes through Conv/Pool and Linear layers.
    Yields prediction labels and prototype embedding vector.
    """
    def __init__(self, num_classes=2):
        super(CPN, self).__init__()
        
        # Input shape: (Batch Size, 1, 5, 5)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # output: (16, 3, 3)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # output: (32, 2, 2)
        
        self.fc1 = nn.Linear(32 * 2 * 2, 64)
        
        # This linear layer produces the embedding vector v(phi; x)
        self.fc2 = nn.Linear(64, 32)
        
        # Branch 1 classification output 
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(32, num_classes)
        
    def forward(self, x):
        """
        Returns:
            logits: Unnormalized network predictions (Batch, num_classes)
            embedding: Low dimensional representative vector v(phi; x) (Batch, 32)
        """
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1) # Flatten
        
        x = F.relu(self.fc1(x))
        
        # Embedding vector v(phi; x) corresponding to Branch 2
        embedding = F.relu(self.fc2(x)) 
        
        # Branch 1 (Dropout -> Linear)
        # Note: We do NOT apply Softmax here. CrossEntropyLoss in PyTorch handles it dynamically.
        # During inference (testing), the paper states we drop this branch entirely
        # and measure Euclidean distance to prototypes instead.
        c = self.dropout(embedding)
        logits = self.classifier(c)
        
        return logits, embedding
