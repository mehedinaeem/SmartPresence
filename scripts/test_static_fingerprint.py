#!/usr/bin/env python3
"""Demonstrate simulated fingerprint verification without launching vision models."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attendance.session_manager import start_session
from src.fingerprint.verification import verify_fingerprint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static fingerprint identity mapping demo; no fingerprint image matching.")
    parser.add_argument("fingerprint_id", nargs="?", help="Simulated scanner ID, FP001–FP040")
    parser.add_argument("--create-session", action="store_true", help="Append an attendance session after successful verification")
    args = parser.parse_args(argv)
    print("============================================\n       SmartPresence Fingerprint Demo\n============================================")
    try:
        fingerprint_id = args.fingerprint_id if args.fingerprint_id is not None else input("Enter fingerprint ID: ")
        result = verify_fingerprint(fingerprint_id)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("\nDemo cancelled.")
        return 1

    print("\nFingerprint Verification\n--------------------------------------------")
    print(f"Fingerprint ID : {result['fingerprint_id']}")
    if not result["verified"]:
        print(f"Result         : NOT VERIFIED\nReason         : {result['reason']}\n--------------------------------------------")
        return 1
    print(f"Template ID    : {result['template_id']}")
    print("Finger         : Right Index")
    print(f"Roll Number    : {result['roll_number']}")
    print(f"Face Status    : {result['face_status'].title()}")
    print("Result         : VERIFIED")
    if args.create_session:
        try:
            session_id = start_session(result["roll_number"], result["fingerprint_id"])
        except OSError as error:
            print(f"Error creating session: {error}", file=sys.stderr)
            return 2
        print(f"Session ID     : {session_id}")
    print("--------------------------------------------\n\nRecommended Visual Path:")
    print("YOLO + ByteTrack" if result["face_status"] == "covered" else "FaceNet + SVM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
