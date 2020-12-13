import torch
import torch.nn as nn


class conv_block(nn.Module):
    """
    Convolution block containing a convolution layer, BN layer and ReLU activation.
    """
    def __init__(self, input_channels, output_channels, kernel_size=5, stride=2, padding=1):
        super().__init__()
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.bn = torch.nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
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
        self.linear_layers = nn.Sequential(*[linear_block(128, 256), linear_block(256, 512), linear_block(512, 1024)])
        self.final_fc = nn.Linear(1024, 784)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_s = x.shape
        x = x.reshape(x_s[0], -1)
        x = self.linear_layers(x)
        x = self.final_fc(x)
        x = self.tanh(x)
        x = x.view(x.shape[0], 1, 28, 28)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layers = nn.Sequential(*[linear_block(784, 1024), linear_block(1024, 512), linear_block(512, 256)])
        self.linear = nn.Linear(256, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear_layers(x)
        x = self.softmax(self.linear(x))
        return x
