from quadratic_autoencoder.utils.QuadraticOperation import *

'''Quadratic Autoencoder'''

class QAE(nn.Module):
    def __init__(self, image_size1: int, image_size2: int):
        super(QAE, self).__init__()
        self.encoder = nn.Sequential(
            QuadraticOperation(image_size1 * image_size2, 512),
            nn.ReLU(),
            QuadraticOperation(512, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            QuadraticOperation(32, 512),
            nn.ReLU(),
            QuadraticOperation(512, image_size1 * image_size2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


if __name__ == '__main__':
    m = QAE(28,28)
    a = torch.randn(10, 28 * 28)
    y, _ = m(a)
    print(y)