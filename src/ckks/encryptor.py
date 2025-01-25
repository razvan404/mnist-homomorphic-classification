from enum import Enum
import base64

import torch
import numpy as np
import tenseal as ts


class PredefinedConfigs(Enum):
    HIGH_DEPTH = {
        "poly_modulus_degree": 16384,
        "coeff_mod_bit_sizes": [31, 26, 26, 26, 26, 26, 26, 31],
    }

    BALANCED_PRECISION = {
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [31, 25, 25, 25, 25, 31],
    }

    HIGH_PRECISION = {
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 40, 40, 60],
    }

    LIGHTWEIGHT = {
        "poly_modulus_degree": 4096,
        "coeff_mod_bit_sizes": [30, 20, 20, 30],
    }


class Encryptor:
    def __init__(
        self,
        config: PredefinedConfigs = None,
        windows_nb: int = 121,
        context: ts.Context | None = None,
    ):
        if context:
            self.context = context
        else:
            if config is None:
                config = PredefinedConfigs.HIGH_DEPTH

            self.context = self.create_ckks_context(**config.value)

        self.windows_nb = windows_nb

    @classmethod
    def create_ckks_context(
        cls, poly_modulus_degree: int, coeff_mod_bit_sizes: list[int]
    ):
        assert len(set(coeff_mod_bit_sizes[1:-1])) == 1
        assert coeff_mod_bit_sizes[0] == coeff_mod_bit_sizes[-1]
        bits_scale = coeff_mod_bit_sizes[1]

        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        )

        context.global_scale = pow(2, bits_scale)
        context.generate_galois_keys()

        return context

    def encrypt_image(self, data: np.ndarray | torch.Tensor) -> ts.CKKSVector:
        assert self.has_secret_key()
        data_enc, windows_nb = ts.im2col_encoding(
            self.context, data.squeeze().tolist(), 7, 7, 2
        )
        assert windows_nb == self.windows_nb
        return data_enc

    def decrypt(self, enc_data: ts.CKKSVector) -> np.ndarray:
        assert self.has_secret_key()
        data = enc_data.decrypt(self.context.secret_key())
        return np.array(data)

    def has_secret_key(self):
        return self.context.has_secret_key()

    @classmethod
    def bytes_to_string(cls, byte_data: bytes) -> str:
        return base64.b64encode(byte_data).decode("utf-8")

    @classmethod
    def string_to_bytes(cls, string_data: str) -> bytes:
        return base64.b64decode(string_data.encode("utf-8"))

    def serialize(self) -> dict:
        return {
            "context": self.bytes_to_string(
                self.context.serialize(save_secret_key=False)
            ),
            "windows_nb": self.windows_nb,
        }

    @classmethod
    def deserialize(cls, serialized_encryptor: dict) -> "Encryptor":
        context = ts.context_from(cls.string_to_bytes(serialized_encryptor["context"]))
        windows_nb = serialized_encryptor["windows_nb"]
        return Encryptor(context=context, windows_nb=windows_nb)

    def serialize_data(self, vec: ts.CKKSVector) -> str:
        vec.link_context(self.context)
        return self.bytes_to_string(vec.serialize())

    def deserialize_data(self, serialized_vec: str) -> ts.CKKSVector:
        return ts.ckks_vector_from(self.context, self.string_to_bytes(serialized_vec))
