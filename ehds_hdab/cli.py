import argparse
import json

from .pipeline import process_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Process EHDS health-data requests")
    parser.add_argument("requests", nargs="+", help="One or more request JSON files")
    args = parser.parse_args(argv)
    responses = [process_file(path) for path in args.requests]
    output = responses[0] if len(responses) == 1 else responses
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
