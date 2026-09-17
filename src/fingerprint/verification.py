"""Simulated fingerprint verification by static identity lookup only."""
from src.fingerprint.fingerprint_database import FingerprintDatabase


def verify_fingerprint(fingerprint_id: str) -> dict:
    """Return a verified identity or an explicit lookup failure; no image matching."""
    return FingerprintDatabase().verify(fingerprint_id)
