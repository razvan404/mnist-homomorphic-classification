import tenseal as ts
import torch


def create_ckks_context():
    bits_scale = 23

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )

    context.global_scale = pow(2, bits_scale)
    context.generate_galois_keys()

    return context


def encrypt_data(context: ts.Context, data: torch.Tensor) -> ts.CKKSVector:
    data_enc, windows_nb = ts.im2col_encoding(
        context, data.squeeze().tolist(), 7, 7, 2
    )
    assert windows_nb == 121
    return data_enc


def decrypt_data(context: ts.Context, enc_data: ts.CKKSVector) -> torch.Tensor:
    data = enc_data.decrypt(context.secret_key())
    return torch.tensor(data)
