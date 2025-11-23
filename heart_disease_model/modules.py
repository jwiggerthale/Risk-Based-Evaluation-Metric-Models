import torch.nn as nn
import torch


class heart_model(nn.Module):
    def __init__(self, 
                 input_dim: int = 21,
                 layers: list = [64, 128, 128, 64],
                 num_classes: int = 2, 
                 temperature: float = 1.5):
        super().__init__()
        self.layers = []
        last_dim = input_dim
        for layer in layers:
            self.layers.append(nn.Linear(last_dim, layer))
            self.layers.append(nn.ReLU())
            last_dim = layer
        self.layers.append(nn.Linear(last_dim, 1))
        # WICHTIG: Kein Sigmoid hier!
        self.model = nn.Sequential(*self.layers)
        self.temperature = temperature #nn.Parameter(torch.tensor(temperature))
    
    def forward(self, x: torch.tensor, 
               inference: bool = False):
        logits = self.model(x)
        # Temperature Scaling VOR Sigmoid
        if inference == True:
            logits = logits / self.temperature
        pred = torch.sigmoid(logits)
        return pred