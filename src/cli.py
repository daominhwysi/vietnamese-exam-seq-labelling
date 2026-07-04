import sys
from pathlib import Path

# Setup local import paths
sys.path.append(str(Path(__file__).parent.parent))

from src.cli import get_parser, execute_command

def main():
    parser = get_parser()
    args = parser.parse_args()
    execute_command(args)

if __name__ == "__main__":
    main()
