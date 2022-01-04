import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input: int, output: int):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input, 1000),
            nn.ReLU(),
            nn.Linear(1000, output),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.fc(x)
        return out