from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os
import numpy as np

# Generate ECC key pair
def generate_keys():
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    return private_key, public_key

# Generate shared AES key using ECDH
def generate_shared_key(private_key, peer_public_key):
    shared_key = private_key.exchange(ec.ECDH(), peer_public_key)

    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,  # AES-256 key size
        salt=None,
        info=b'fl-iot'
    ).derive(shared_key)

    return derived_key

# AES Encryption
def encrypt(data, key):

    # convert numpy array to bytes
    data_bytes = data.tobytes()

    # generate random IV
    iv = os.urandom(16)

    # AES Cipher
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()

    # padding
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data_bytes) + padder.finalize()

    # encrypt data
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    return encrypted_data, iv

# AES Decryption
def decrypt(encrypted_data, key, iv, shape, dtype=np.float32):
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()

    decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()

    unpadder = padding.PKCS7(128).unpadder()
    decrypted_data = unpadder.update(decrypted_padded) + unpadder.finalize()

    return np.frombuffer(
        decrypted_data,
        dtype=dtype
    ).reshape(shape)









# from cryptography.hazmat.primitives.asymmetric import ec
# from cryptography.hazmat.primitives import hashes
# from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# # Generate ECC key pair
# def generate_keys():
#     private_key = ec.generate_private_key(ec.SECP256R1())
#     public_key = private_key.public_key()
#     return private_key, public_key

# # Generate shared key using ECDH
# def generate_shared_key(private_key, peer_public_key):
#     shared_key = private_key.exchange(ec.ECDH(), peer_public_key)

#     derived_key = HKDF(
#         algorithm=hashes.SHA256(),
#         length=16,
#         salt=None,
#         info=b'fl-iot'
#     ).derive(shared_key)

#     # convert to simple integer key (for lightweight encryption)
#     return int.from_bytes(derived_key, 'big') % 1000

# # Lightweight encryption (for model weights)
# def encrypt(data, key):
#     return data + key

# def decrypt(data, key):
#     return data - key



