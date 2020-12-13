import torch
import torch.nn as nn


class conv_block(nn.Module):
    """
    Convolution block containing a convolution layer, ReLU activation and Batchnorm layer.
    """
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, output_padding=1, upsample=False):
        super().__init__()
        if upsample:
            self.conv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding,
                                           output_padding=output_padding)
        else:
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.bn = torch.nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class linear_block(nn.Module):
    """
    Linear/FC layer followed by BN and RelU activation.
    """
    def __init__(self, input_features, output_features):
        super().__init__()
        self.fc = nn.Linear(input_features, output_features)
        self.bn = torch.nn.BatchNorm1d(output_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 12544)
        self.conv_layers = nn.Sequential(*[conv_block(256, 128, stride=2, upsample=True), 
                                           conv_block(128, 64, output_padding=0, upsample=True), 
                                           conv_block(64, 32, output_padding=0, upsample=True), 
                                           conv_block(32, 1, stride=2, upsample=True)])
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], -1, 7, 7)
        x = self.conv_layers(x)
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(*[conv_block(1, 32, stride=2), 
                                           conv_block(32, 64), 
                                           conv_block(64, 128),
                                           conv_block(128, 256, stride=2)])
        self.linear = nn.Linear(12544, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, 28, 28)
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x[:,0]
