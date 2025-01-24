import tqdm
import torch
import tenseal as ts

from src.classifier import MnistClassifier
from src.ckks.ckks_classifier import CkksCompatibleMnistClassifier
from src.ckks.utils import create_ckks_context, encrypt_data, decrypt_data


def test_ckks_classifier(context: ts.Context):
    classifier = MnistClassifier()
    classifier.eval()
    enc_classifier = CkksCompatibleMnistClassifier(classifier, windows_nb=121)

    batch_example = torch.randint(0, 255, (4, 1, 28, 28)) / 255.
    for input_example in tqdm.tqdm(batch_example):
        input_enc = encrypt_data(context, input_example)
        output_enc = enc_classifier(input_enc)
        output = decrypt_data(context, output_enc)
        assert output.shape == (10,)


if __name__ == "__main__":
    context = create_ckks_context()
    test_ckks_classifier(context)