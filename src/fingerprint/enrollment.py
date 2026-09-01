from src.fingerprint.fingerprint_database import FingerprintDatabase


def enrollment_record(roll_number: str) -> dict[str, str]:
    fingerprint_id = FingerprintDatabase().fingerprint_for(roll_number)
    if fingerprint_id is None:
        raise KeyError(f"Unknown active roll number: {roll_number}")
    return {"roll_number": str(roll_number), "fingerprint_id": fingerprint_id}

