import torch.nn as nn
import torch
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
        self.fc = nn.Sequential(nn.Linear(1000,2))
    def forward(self, x):
        output = self.model(x)
        output = self.fc(output)
        return output
    def get_embedding(self,x):
        return self.forward(x)