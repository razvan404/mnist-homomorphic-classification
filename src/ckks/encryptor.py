from enum import Enum

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
    def __init__(self, config: PredefinedConfigs = None, windows_nb: int = 121):
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

    def encrypt(self, data: np.ndarray) -> ts.CKKSVector:
        assert self.context.has_secret_key()
        data_enc, windows_nb = ts.im2col_encoding(
            self.context, data.squeeze().tolist(), 7, 7, 2
        )
        assert windows_nb == self.windows_nb
        print()
        return data_enc

    def decrypt(self, enc_data: ts.CKKSVector) -> np.ndarray:
        assert self.context.has_secret_key()
        data = enc_data.decrypt(self.context.secret_key())
        return np.array(data)
