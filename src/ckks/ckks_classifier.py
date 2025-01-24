import torch
import tenseal as ts

from src.classifier import MnistClassifier


class CkksCompatibleMnistClassifier:
    def __init__(self, model: MnistClassifier, windows_nb: int = 121):
        self.conv_weight = model.conv.weight.data.view(
            model.conv.out_channels, *model.conv.kernel_size
        ).tolist()
        self.conv_bias = model.conv.bias.data.tolist()

        self.fc1_weight = model.fc1.weight.T.data.tolist()
        self.fc1_bias = model.fc1.bias.data.tolist()

        self.fc2_weight = model.fc2.weight.T.data.tolist()
        self.fc2_bias = model.fc2.bias.data.tolist()

        self.windows_nb = windows_nb

    @torch.no_grad()
    def forward(self, enc_x: ts.CKKSVector):
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv_weight, self.conv_bias):
            y = enc_x.conv2d_im2col(kernel, self.windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        # square activation
        enc_x.square_()
        # fc2 layer
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
