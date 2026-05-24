from hashlib import sha256


def extract_words(description: str) -> list[str]:
    """Extract alphanumeric words from description for key generation."""
    cleaned = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in description)
    words = [w.lower() for w in cleaned.split() if w]
    return words


def hash_hex(text: str, n_bytes: int = 2) -> str:
    """Returns `n_bytes * 2` hexadecimal chars of the sha256 of the text."""
    return sha256(text.encode('utf-8')).digest()[:n_bytes].hex()


def get_memory_key(description: str) -> str:
    words = extract_words(description)
    slug = '_'.join(words[:4])
    checksum = hash_hex(description)
    return f"task_{slug}_{checksum}_notes"
