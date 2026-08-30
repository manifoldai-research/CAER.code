#!/usr/bin/env python3
import argparse
import os
import selectors
import sys
import time


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _append_closed(path, data):
    if not data:
        return
    with open(path, "ab", buffering=0) as handle:
        handle.write(data)


def main():
    parser = argparse.ArgumentParser(
        description="Mirror stdin to stdout immediately and append closed chunks to a shared log."
    )
    parser.add_argument("log_path")
    parser.add_argument("--flush-seconds", type=float, default=2.0)
    parser.add_argument("--max-buffer-bytes", type=int, default=1024 * 1024)
    args = parser.parse_args()

    log_path = os.path.abspath(args.log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    flush_seconds = max(float(args.flush_seconds), 0.1)
    max_buffer_bytes = max(int(args.max_buffer_bytes), 4096)

    selector = selectors.DefaultSelector()
    selector.register(sys.stdin.buffer, selectors.EVENT_READ)
    pending = bytearray()
    next_flush = time.monotonic() + flush_seconds

    try:
        while True:
            timeout = max(next_flush - time.monotonic(), 0.0)
            events = selector.select(timeout)
            if events:
                chunk = os.read(sys.stdin.fileno(), 65536)
                if not chunk:
                    _append_closed(log_path, pending)
                    pending.clear()
                    return
                _write_all(sys.stdout.fileno(), chunk)
                pending.extend(chunk)

            now = time.monotonic()
            if pending and (now >= next_flush or len(pending) >= max_buffer_bytes):
                _append_closed(log_path, pending)
                pending.clear()
                next_flush = now + flush_seconds
            elif now >= next_flush:
                next_flush = now + flush_seconds
    finally:
        if pending:
            _append_closed(log_path, pending)


if __name__ == "__main__":
    main()
