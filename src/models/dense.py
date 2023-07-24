from torch import nn


class LinearAE(nn.Module):
    def __init__(self, img_size, img_channels, hidden_size=256):
        super().__init__()
        input_size = img_size ** 2 * img_channels
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Unflatten(1, (img_channels, img_size, img_size)),
            nn.Sigmoid())

    def forward(self, x):
        emb = self.encoder(x)
        out = self.decoder(emb)
        return emb, out


class DenseAE(nn.Module):
    def __init__(self, img_size, img_channels, hidden_size=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size ** 2 * img_channels, 2 * hidden_size),
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_size, img_size ** 2 * img_channels),
            nn.Unflatten(1, (img_channels, img_size, img_size)),
            nn.Sigmoid())

    def forward(self, x):
        emb = self.encoder(x)
        out = self.decoder(emb)
        return emb, out
