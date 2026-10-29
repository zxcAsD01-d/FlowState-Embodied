"""Run the AL-Duet platform: python -m backend [--port 8830]."""
from __future__ import annotations
import argparse
import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description="AL-Duet platform")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8830)
    ap.add_argument("--log-level", default="warning")
    args = ap.parse_args()
    uvicorn.run("backend.app:app", host=args.host, port=args.port,
                log_level=args.log_level, reload=False)


if __name__ == "__main__":
    main()
