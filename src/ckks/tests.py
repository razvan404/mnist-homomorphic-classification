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
        input_enc = encryptor.encrypt_image(input_example)
        output_enc = enc_classifier(input_enc)
        output = encryptor.decrypt(output_enc)
        assert output.shape == (10,)


def test_encoder_serializer(encryptor: Encryptor):
    random_img = torch.rand(28, 28)
    enc_img = encryptor.encrypt_image(random_img)

    serialized_enc_img = encryptor.serialize_data(enc_img)
    serialized_encryptor = encryptor.serialize()

    public_deserialized_encryptor = Encryptor.deserialize(serialized_encryptor)
    assert public_deserialized_encryptor.has_secret_key() is False

    deserialized_enc_img = public_deserialized_encryptor.deserialize_data(
        serialized_enc_img
    )
    assert serialized_enc_img == public_deserialized_encryptor.serialize_data(
        deserialized_enc_img
    )


if __name__ == "__main__":
    encryptor = Encryptor()
    test_ckks_classifier(encryptor)
    print("CKKS Classifier Test ✅")
    test_encoder_serializer(encryptor)
    print("Encoder Serializer Test ✅")
