#!/usr/bin/env python3
"""
ucinet engine implementation

- it uses the chess library to handle the chess logic
- it implements a basic UCI protocol
- it can communicate via named pipes or stdin/stdout
  - stdin/stdout is the standard way UCI engines communicate
- named pipes are useful for debugging and development
  - it uses the `named_pipe_forwarder.py` script to forward stdin/stdout to the named pipes

## Why named pipes?
This is a trick to help during development. UCI protocol is above stdin/stdout.
This means the UCI client (aka the GUI) will open a subprocess and communicate
with it via stdin/stdout.

This make is hard to debug the engine.

### Reasons
- hard to relaunch the engine
- hard to log the communication between the GUI and the engine
- hard to launch the server in debug mode in vscode

### Solution
- have the engine communicate via named pipes instead of stdin/stdout
- have a small forwarder script that read stdin and write to the named pipe
    and read from the other named pipe and write to stdout. `./named_pipe_forwarder.py`
- have the GUI to talk to the forwarder script instead of the engine directly
"""

# stdlib imports
import sys
from typing import Optional, Callable
import os
import time

# pip imports
import argparse
import chess
import chess.polyglot

# local imports
from src.libs.chess_player import ChessPlayer
from src.utils.model_utils import ModelUtils

# setup __dirname__
__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "../data")
output_folder_path = os.path.join(__dirname__, "..", "output")
model_folder_path = os.path.join(output_folder_path, "model")

proto_in_path = os.path.join(output_folder_path, "ucinet_proto_in")
proto_out_path = os.path.join(output_folder_path, "ucinet_proto_out")
log_file_path = os.path.join(output_folder_path, "ucinet_engine.log")


# =============================================================================
# UCI Engine implementation
# =============================================================================
class UciNetEngine:
    def __init__(self, ignore_quit: bool = False):
        self._ignore_quit = ignore_quit

        self._current_board = chess.Board()

        # Load the model
        model_name = ModelUtils.MODEL_NAME.CHESS_MODEL_CONV2D
        self._model = ModelUtils.load_model(model_name, model_folder_path)

        # Read the polyglot opening book
        polyglot_path = os.path.join(data_folder_path, "./polyglot/lichess_pro_books/lpb-allbook.bin")
        self._polyglot_reader = chess.polyglot.open_reader(polyglot_path)

    def _log(self, msg: str) -> None:
        """Append a timestamped message to the log file."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"{ts} {msg}\n"
        try:
            with open(log_file_path, "a", encoding="utf-8", errors="replace") as lf:
                lf.write(line)
        except Exception:
            # Never raise from logging; fall back to stderr
            print(f"[log error] could not write to ucinet_engine.log", file=sys.stderr)

    def process_ucinet_command(self, input_cmd: str) -> str:
        output_cmd: str = ""
        if input_cmd == "uci":
            # =============================================================================
            # "uci" command is sent to the engine to identify itself and its capabilities.
            # =============================================================================
            output_cmd += "id name chess_ai.ml\n"
            output_cmd += "id author jerome Etienne\n"
            output_cmd += "uciok"
        elif input_cmd == "isready":
            # =============================================================================
            # "isready" command is sent to the engine to check if it is ready to accept commands.
            # =============================================================================
            output_cmd = "readyok"
        elif input_cmd.startswith("position"):
            # =============================================================================
            # "position" command is sent to the engine to set up the board position.
            # =============================================================================
            # input_cmd is like "position fen rnbqkb1r/pppp1ppp/4pn2/8/P2P4/4P3/1PP2PPP/RNBQKBNR b KQkq a3 0 3

            # split into words
            words = input_cmd.split(" ")
            #
            if len(words) >= 3 and words[1] == "fen":
                fen = " ".join(words[2:])
                self._current_board = chess.Board(fen=fen)
                self._log(f"Set board to FEN: {fen}")
            else:
                raise ValueError(f"Unsupported position command format: {input_cmd}")
        elif input_cmd.startswith("go"):
            # =============================================================================
            # "go" command is sent to the engine to start calculating the best move.
            # =============================================================================
            # initialize chess player
            player_color = self._current_board.turn
            chess_player = ChessPlayer(self._model, player_color, self._polyglot_reader)
            # predict the best move
            move_uci = chess_player.predict_next_move(self._current_board)
            if move_uci is not None:
                # play the move on the internal board
                move = chess.Move.from_uci(move_uci)
                self._current_board.push(move)
                # log the event
                self._log(f"Playing move: {move_uci}")
                # respond with the best move
                output_cmd += f"bestmove {move_uci}"
            else:
                self._log("No legal moves available, game over?")
                output_cmd += "bestmove (none)"
        elif input_cmd == "ucinewgame":
            # =============================================================================
            # "ucinewgame" command is sent to the engine when a new game is started.
            # =============================================================================
            self._current_board.reset()
        elif input_cmd == "stop":
            # =============================================================================
            # "stop" command is sent to the engine to stop calculating the best move.
            # =============================================================================
            pass
        elif input_cmd == "quit":
            # =============================================================================
            # "quit" command is sent to the engine to tell it to exit.
            # =============================================================================
            if self._ignore_quit is False:
                self._log("Received 'quit' command, exiting...")
                sys.exit(0)
        else:
            self._log(f"Unknown command: {input_cmd}")
            output_cmd += ""  # Ignore unsupported commands for now

        return output_cmd


# =============================================================================
# Server implementations
# =============================================================================
class UciNetServerNamedPipe:
    """
    A ucinet server that communicates via named pipes.
    """

    def __init__(self):
        self._proto_out = open(proto_out_path, "wb", buffering=0)

    def serve(self, process_cmd_func: Callable[[str], str]) -> None:
        """
        Serve the named pipe, using the provided callback(str) -> str to process each input line.
        """
        try:
            while True:
                # Open for reading; this will block until a writer opens the FIFO.
                try:
                    with open(proto_in_path, "rb", buffering=0) as proto_in:
                        while True:
                            line = proto_in.readline()
                            if not line:
                                # Writer closed -> EOF. Break to reopen and wait for
                                # the next writer instead of exiting the process.
                                print("[ucinet_engine] proto_in EOF, reopening...", file=sys.stderr)
                                break

                            line_text = line.decode("utf-8").strip()
                            print(f"Received line: {line_text}", file=sys.stderr)

                            output_cmd = process_cmd_func(line_text)
                            output_cmd += "\n"
                            print(f"Sending line: {output_cmd}", file=sys.stderr)
                            if output_cmd:
                                self._proto_out.write(output_cmd.encode("utf-8"))
                                self._proto_out.flush()
                except FileNotFoundError:
                    print(f"[ucinet_engine] FIFO '{proto_in_path}' not found, retrying...", file=sys.stderr)
                    time.sleep(1.0)
                    continue

        except KeyboardInterrupt:
            # Allow graceful exit on Ctrl-C
            pass


class UciNetServerStdinStdout:
    """
    A ucinet server that communicates via stdin/stdout.

    This is the standard way UCI engines communicate.
    """

    def __init__(self):
        pass

    def serve(self, process_cmd_func: Callable[[str], str]) -> None:
        """
        Serve stdin/stdout, using the provided callback(str) -> str to process each input line.
        """
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    # EOF
                    print("[ucinet_engine] stdin EOF, exiting...", file=sys.stderr)
                    break

                line_text = line.strip()
                print(f"Received line: {line_text}", file=sys.stderr)

                output_cmd = process_cmd_func(line_text)
                output_cmd += "\n"
                print(f"Sending line: {output_cmd}", file=sys.stderr)
                if output_cmd:
                    print(f"Sending response: {output_cmd}", file=sys.stderr)
                    sys.stdout.write(output_cmd)
                    sys.stdout.flush()

        except KeyboardInterrupt:
            # Allow graceful exit on Ctrl-C
            pass


# =============================================================================
# Main function
# =============================================================================
def main() -> None:

    # =============================================================================
    # Argument parsing
    # =============================================================================
    argParser = argparse.ArgumentParser(description="UCI chess engine over named pipes or stdin/stdout")
    argParser.add_argument(
        "--mode",
        "-m",
        choices=["named_pipe", "stdio"],
        default="stdio",
        help="Communication mode: 'named_pipe' or 'stdio' (default)",
    )
    args = argParser.parse_args()

    # =============================================================================
    # Setup server and engine
    # =============================================================================
    # create engine
    ignore_quit = True if args.mode == "named_pipe" else False
    engine = UciNetEngine(ignore_quit=ignore_quit)

    # create server
    if args.mode == "named_pipe":
        server = UciNetServerNamedPipe()
    elif args.mode == "stdio":
        server = UciNetServerStdinStdout()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # =============================================================================
    # Run server
    # =============================================================================

    # start serving
    server.serve(engine.process_ucinet_command)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    main()
