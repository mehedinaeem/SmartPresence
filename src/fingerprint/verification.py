from src.fingerprint.fingerprint_database import FingerprintDatabase


def verify_fingerprint(fingerprint_id: str) -> str | None:
    """Resolve a scanner-produced template ID; hardware matching stays in the device adapter."""
    return FingerprintDatabase().roll_for(fingerprint_id)

