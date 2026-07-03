import hashlib
from pathlib import Path

def get_file_hash(path: Path) -> str:
    """Computes the MD5 hash of a file to check for exact duplicates."""
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()
