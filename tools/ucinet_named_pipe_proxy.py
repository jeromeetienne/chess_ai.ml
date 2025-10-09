#!/usr/bin/env python3
"""
named_pipe_forwarder.py

Read lines from stdin and forward each line into the FIFO at ./proto_in.
Simultaneously read lines from the FIFO at ./proto_out and write them to stdout
as soon as they arrive.

The script is resilient to the other side opening/closing the FIFOs: it will
reopen on writer/reader disconnects and retry when the peer is not present.
"""

from __future__ import annotations

import errno
import os
import signal
import stat
import sys
import threading
import time
from datetime import datetime

__dirname__ = os.path.dirname(os.path.abspath(__file__))

# Log file path (placed next to this script). Use './named_pipe.log' semantics.
LOG_PATH = os.path.join(__dirname__, "../output/ucinet_named_pipe.log")
PROTO_IN = f"{__dirname__}/../output/ucinet_proto_in"
PROTO_OUT = f"{__dirname__}/../output/ucinet_proto_out"


log_lock = threading.Lock()


def log(msg: str, to_stderr: bool = False) -> None:
    """Append a timestamped message to the log file in a thread-safe way.

    If `to_stderr` is True, also print the message to stderr. This replaces
    existing call sites that separately printed to stderr and then called
    `log()`.
    """
    if to_stderr:
        # Mirror previous behaviour: print the raw message to stderr
        print(msg, file=sys.stderr)

    ts = datetime.now().isoformat(sep=" ", timespec="seconds")
    line = f"{ts} {msg}\n"
    try:
        with log_lock:
            with open(LOG_PATH, "a", encoding="utf-8", errors="replace") as lf:
                lf.write(line)
    except Exception:
        # Never raise from logging; fall back to stderr
        print(f"[log error] could not write to {LOG_PATH}", file=sys.stderr)


stop_event = threading.Event()


def is_fifo(path: str) -> bool:
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISFIFO(st.st_mode)


def read_fifo_to_stdout(path: str) -> None:
    """Continuously read lines from FIFO at `path` and print them to stdout.

    If the writer closes the FIFO, this function will loop and reopen it when
    a new writer appears.
    """
    log(f"reader thread starting for {path}")
    while not stop_event.is_set():
        if not is_fifo(path):
            msg = f"[reader] FIFO '{path}' not found, retrying..."
            log(msg, to_stderr=True)
            time.sleep(1.0)
            continue

        try:
            # Opening for read will block until a writer opens the FIFO.
            with open(path, "r", encoding="utf-8", errors="replace") as rf:
                log(f"reader opened {path}")
                for line in rf:
                    # print immediately and flush
                    print(line, end="", flush=True)
                    log(f"reader received: {line.rstrip()}")
                    if stop_event.is_set():
                        break
                # when the writer closes, the for-loop ends; reopen the FIFO
                log(f"reader detected writer closed on {path}")
        except Exception as e:
            log(f"[reader] error reading '{path}': {e}", to_stderr=True)
            time.sleep(0.5)


def stdin_to_fifo(path: str) -> None:
    """Read stdin lines and write each line to the FIFO at `path`.

    Each line will be written by opening the FIFO, writing the line, flushing
    and closing the FIFO. If no reader is present, the open for writing would
    normally block; instead we use non-blocking opens and retry until a
    reader appears (so the script can continue to accept stdin without
    permanently blocking).
    """
    # Iterate over stdin lines (this blocks waiting for input)
    log(f"writer thread starting for {path}")
    for line in sys.stdin:
        if stop_event.is_set():
            break

        log(f"stdin line: {line.rstrip()}")

        # Keep trying to open for writing until we succeed or are asked to stop.
        while not stop_event.is_set():
            try:
                # Use os.open with O_NONBLOCK so that we can detect "no reader"
                fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            except FileNotFoundError:
                msg = f"[writer] FIFO '{path}' not found, retrying..."
                log(msg, to_stderr=True)
                time.sleep(1.0)
                continue
            except OSError as e:
                # ENXIO: No reader on the FIFO
                if e.errno == errno.ENXIO:
                    # Wait a short while for a reader to appear
                    time.sleep(0.1)
                    continue
                else:
                    msg = f"[writer] error opening '{path}' for write: {e}"
                    log(msg, to_stderr=True)
                    time.sleep(0.5)
                    continue

            # We have a valid fd that can be written to. Wrap and write.
            try:
                with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as wf:
                    wf.write(line)
                    wf.flush()
                log(f"writer wrote to {path}: {line.rstrip()}")
                break
            except BrokenPipeError:
                # Reader closed between open and write; try again
                log(f"writer BrokenPipe when writing to {path}")
                time.sleep(0.05)
                continue
            except Exception as e:
                msg = f"[writer] error writing to '{path}': {e}"
                log(msg, to_stderr=True)
                # Avoid tight loop on unknown errors
                time.sleep(0.2)
                break


def handle_signals(signum, frame) -> None:
    stop_event.set()
    log(f"signal {signum} received, stopping")


def main() -> int:
    # register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_signals)
    signal.signal(signal.SIGTERM, handle_signals)

    # Check that the FIFOs exist, if not create it
    if not os.path.exists(PROTO_IN):
        try:
            os.mkfifo(PROTO_IN)
            log(f"Created FIFO {PROTO_IN}")
        except Exception as e:
            msg = f"Error creating FIFO '{PROTO_IN}': {e}"
            log(msg, to_stderr=True)
            return 1
    if not os.path.exists(PROTO_OUT):
        try:
            os.mkfifo(PROTO_OUT)
            log(f"Created FIFO {PROTO_OUT}")
        except Exception as e:
            msg = f"Error creating FIFO '{PROTO_OUT}': {e}"
            log(msg, to_stderr=True)
            return 1

    if not is_fifo(PROTO_IN):
        msg = f"Warning: '{PROTO_IN}' does not exist or is not a FIFO."
        log(msg, to_stderr=True)
    if not is_fifo(PROTO_OUT):
        msg = f"Warning: '{PROTO_OUT}' does not exist or is not a FIFO."
        log(msg, to_stderr=True)

    log("named_pipe_forwarder main starting")

    reader = threading.Thread(target=read_fifo_to_stdout, args=(PROTO_OUT,), daemon=True)
    writer = threading.Thread(target=stdin_to_fifo, args=(PROTO_IN,), daemon=True)

    reader.start()
    writer.start()

    # Wait until writer thread finishes (stdin EOF) or stop requested.
    try:
        while writer.is_alive() and not stop_event.is_set():
            writer.join(timeout=0.1)
    except KeyboardInterrupt:
        stop_event.set()
        log("KeyboardInterrupt received, stopping")

    # allow reader to finish current work, then exit
    stop_event.set()
    reader.join(timeout=1.0)

    log("named_pipe_forwarder exiting")

    return 0


if __name__ == "__main__":
    status_code = main()
    sys.exit(status_code)
