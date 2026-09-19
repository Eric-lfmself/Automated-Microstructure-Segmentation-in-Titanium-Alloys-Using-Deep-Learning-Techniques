"""Print the effective configuration without loading data or model weights."""
import argparse
import json
from configs import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Optional JSON configuration; defaults to the built-in CPU dry-run")
    args = parser.parse_args()
    print(json.dumps(load_config(args.config).to_dict(), indent=2))

if __name__ == "__main__":
    main()
