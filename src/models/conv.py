from torch import nn


def to_activation(name):
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'leaky-relu':
        return nn.LeakyReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(name)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, kernel_size=3):
        super().__init__()
        if kernel_size == 3:
            stride = 2
            padding = 1
        elif kernel_size == 5:
            stride = 4
            padding = 2
        else:
            raise ValueError()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            to_activation(activation)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, kernel_size=3,
                 bn=True):
        super().__init__()
        if kernel_size == 3:
            stride = 2
            padding = 1
            output_padding = 1
        elif kernel_size == 5:
            stride = 4
            padding = 2
            output_padding = 3
        else:
            raise ValueError()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                     stride, padding, output_padding)]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(to_activation(activation))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvAE(nn.Module):
    """
    https://github.com/izikgo/AnomalyDetectionTransformations/blob/master/models/encoders_decoders.py
    """

    def __init__(self, img_size, img_channels,
                 hidden_size=256,
                 num_features=64,
                 kernel_size=3,
                 num_layers=4,
                 activation='relu',
                 hidden_activation='tanh',
                 output_activation='sigmoid'):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = kernel_size - 1
        self.num_layers = num_layers
        self.activation = activation
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        channels = [num_features * 2 ** i for i in range(num_layers)]
        self.encoder = self.to_encoder(channels)
        self.decoder = self.to_decoder(channels)

    def to_encoder(self, channels):
        channels = [self.img_channels] + channels
        blocks = []
        for i in range(self.num_layers):
            blocks.append(EncoderBlock(channels[i], channels[i + 1],
                                       self.activation, self.kernel_size))

        dim = self.img_size // self.stride ** self.num_layers
        return nn.Sequential(
            *blocks,
            nn.Flatten(),
            nn.Linear(channels[-1] * dim ** 2, self.hidden_size),
            to_activation(self.hidden_activation),
        )

    def to_decoder(self, channels):
        channels = channels[::-1] + [self.img_channels]
        blocks = []
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                bn = True
                act = self.activation
            else:
                bn = False
                act = self.output_activation
            blocks.append(DecoderBlock(channels[i], channels[i + 1], act,
                                       self.kernel_size, bn))

        dim = self.img_size // self.stride ** self.num_layers
        return nn.Sequential(
            nn.Linear(self.hidden_size, channels[0] * dim ** 2),
            nn.BatchNorm1d(channels[0] * dim ** 2),
            to_activation(self.activation),
            nn.Unflatten(1, (channels[0], dim, dim)),
            *blocks,
        )

    def forward(self, x):
        emb = self.encoder(x)
        out = self.decoder(emb)
        return emb, out
