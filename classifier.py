import torch
import torch.nn as nn


class MnistClassifier(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding="same"),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 128),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = x ** 2  # Polynomial approximation for non-linearity
        x = self.classifier(x)
        return x


def test_classifier():
    classifier = MnistClassifier(in_channels=1)
    input_example = torch.randn(4, 1, 28, 28)
    output = classifier(input_example)
    assert output.shape == (4, 10)


if __name__ == "__main__":
    test_classifier()