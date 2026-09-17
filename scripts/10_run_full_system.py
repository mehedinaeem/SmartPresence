import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fingerprint.verification import verify_fingerprint
from src.attendance.session_manager import start_session

parser=argparse.ArgumentParser(); parser.add_argument("fingerprint_id")
if __name__ == "__main__":
    args=parser.parse_args(); result=verify_fingerprint(args.fingerprint_id)
    if not result["verified"]: raise SystemExit("Fingerprint verification failed")
    roll = result["roll_number"]
    print({"session_id":start_session(roll,result["fingerprint_id"]),"roll_number":roll})
