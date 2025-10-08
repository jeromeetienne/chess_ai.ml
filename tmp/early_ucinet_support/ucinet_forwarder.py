#!/usr/bin/env python3
"""
Read stdin line-by-line and print each line as soon as it's received.

This uses binary reads/writes and explicit flush to avoid stdio buffering
delays so lines appear immediately when the producer writes them.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    # Use the binary interfaces to avoid extra decoding/encoding buffering
    # and call flush() after every write so output is emitted immediately.
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    # Prepare a log file next to this script. Include a startup timestamp
    # in the filename so multiple runs don't collide.
    script_dir = Path(__file__).resolve().parent
    log_path = script_dir / f"read_lines_unbuffered.log"

    # Open the file in binary append mode. We'll write timestamped lines to it.
    try:
        logfile = open(log_path, "ab")
    except OSError:
        # Fall back to stdout-only if we can't open the file
        logfile = None

    try:
        while True:
            line = stdin.readline()
            if not line:
                # EOF
                break

            # # Echo the raw line to stdout immediately
            # stdout.write(line)
            # stdout.flush()
            line_text = line.decode("utf-8").strip()
            print(f"Received line: {line_text}", file=sys.stderr)
            if line_text == "uci":
                stdout.write("id name chess_ai.ml\n".encode("utf-8"))
                stdout.write("id author jerome Etienne\n".encode("utf-8"))
                stdout.write("uciok\n".encode("utf-8"))
                stdout.flush()
            elif line_text == "isready":
                stdout.write("readyok\n".encode("utf-8"))
                stdout.flush()
            elif line_text.startswith("position"):

            # Write a timestamped copy to the logfile (if available).
            if logfile is not None:
                try:

                    logfile.write(line)
                    logfile.flush()
                except OSError:
                    # If writing fails, drop logging but keep echoing to stdout
                    try:
                        logfile.close()
                    except Exception:
                        pass
                    logfile = None
    except KeyboardInterrupt:
        # Allow graceful exit on Ctrl-C
        pass
    finally:
        if logfile is not None:
            try:
                logfile.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
