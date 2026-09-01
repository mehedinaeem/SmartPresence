"""SmartPresence command dispatcher."""
import argparse
from src.utils.paths import settings

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="SmartPresence research system")
    parser.add_argument("--show-config",action="store_true")
    args=parser.parse_args()
    if args.show_config: print(settings())
    else: parser.print_help()
