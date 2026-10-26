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
    parser.add_argument("--model", choices=["svm", "openai", "local_llm"])
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--payload", type=Path, default=Path(__file__).with_name("predict.json")
    )
    args = parser.parse_args()
    body = json.loads(args.payload.read_text())
    if args.model is not None:
        body["model"] = args.model
    payload = json.dumps(body).encode()
    request = Request(
        args.url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
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
