"""Send a JSON payload to the local API using the Python standard library."""

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main() -> None:
    """Print the API response or an actionable HTTP/connection error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument(
        "--payload", type=Path, default=Path(__file__).with_name("predict.json")
    )
    args = parser.parse_args()
    payload = json.dumps(json.loads(args.payload.read_text())).encode()
    request = Request(
        args.url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            print(json.dumps(json.load(response), indent=2))
    except HTTPError as error:
        parser.exit(1, f"HTTP {error.code}: {error.read().decode()}\n")
    except URLError as error:
        parser.exit(
            1,
            f"Cannot reach {args.url}: {error.reason}. Start the API with make api.\n",
        )


if __name__ == "__main__":
    main()
