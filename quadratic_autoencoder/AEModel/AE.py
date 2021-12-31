import torch.nn as nn
'''Vanilla Autoencoder'''
class AutoEncoder(nn.Module):
    def __init__(self, image_size1: int, image_size2: int):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # Orign AE
            nn.Linear(image_size1 * image_size2, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 512),
            nn.ReLU(),
            nn.Linear(512, image_size1 * image_size2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded