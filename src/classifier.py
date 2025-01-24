import torch
import torch.nn as nn


class MnistClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 64, output_dim: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=7, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(968, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x * x  # square activation function
        x = self.flatten(x)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x


def test_classifier():
    classifier = MnistClassifier()
    input_example = torch.randn(4, 1, 28, 28)
    output = classifier(input_example)
    assert output.shape == (4, 10)


if __name__ == "__main__":
    test_classifier()