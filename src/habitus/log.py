"""Minimal timestamped logger (stderr). No third-party imports."""

import sys
from datetime import datetime


def log(msg, tag=None):
    """Print `<timestamp> - [tag] msg` to stderr and flush."""
    head = f"{datetime.now():%Y-%m-%d %H:%M:%S} - "
    if tag:
        head += f"[{tag}] "
    print(head + str(msg), file=sys.stderr, flush=True)
