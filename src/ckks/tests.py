import tqdm
import torch

from src.classifier import MnistClassifier
from src.ckks.ckks_classifier import CkksCompatibleMnistClassifier
from src.ckks.encryptor import Encryptor


def test_ckks_classifier(encryptor: Encryptor):
    classifier = MnistClassifier()
    classifier.eval()
    enc_classifier = CkksCompatibleMnistClassifier(classifier, windows_nb=121)

    batch_example = torch.randint(0, 255, (4, 1, 28, 28)) / 255.0
    for input_example in tqdm.tqdm(batch_example):
        input_enc = encryptor.encrypt(input_example)
        output_enc = enc_classifier(input_enc)
        output = encryptor.decrypt(output_enc)
        assert output.shape == (10,)


if __name__ == "__main__":
    encryptor = Encryptor()
    test_ckks_classifier(encryptor)
