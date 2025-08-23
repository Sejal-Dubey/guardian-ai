# shared/crypto.py
from cryptography.fernet import Fernet
from shared.config import get_settings

_fernet = Fernet(get_settings().enc_key.encode())

def encrypt_token(token: str) -> str:
    return _fernet.encrypt(token.encode()).decode()

def decrypt_token(blob: str) -> str:
    return _fernet.decrypt(blob.encode()).decode()