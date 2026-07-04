from src.cli.parser import get_parser
from src.cli.commands import execute_command

def main():
    parser = get_parser()
    args = parser.parse_args()
    execute_command(args)

