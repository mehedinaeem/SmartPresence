import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fingerprint.verification import verify_fingerprint
from src.attendance.session_manager import start_session

parser=argparse.ArgumentParser(); parser.add_argument("fingerprint_id")
if __name__ == "__main__":
    args=parser.parse_args(); roll=verify_fingerprint(args.fingerprint_id)
    if roll is None: raise SystemExit("Fingerprint verification failed")
    print({"session_id":start_session(roll,args.fingerprint_id),"roll_number":roll})
