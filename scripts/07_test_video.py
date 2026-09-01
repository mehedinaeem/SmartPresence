import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.video.video_processor import process_video

parser=argparse.ArgumentParser(); parser.add_argument("video"); parser.add_argument("--mode",choices=("experiment","realtime"),default="experiment")
if __name__ == "__main__":
    args=parser.parse_args(); print({"sampled_frames":process_video(args.video,lambda frame,index,timestamp: None,args.mode)})
