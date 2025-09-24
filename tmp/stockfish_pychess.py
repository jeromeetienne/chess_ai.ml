import asyncio
import typing
import chess
import chess.engine
from stockfish import Stockfish

async def main() -> None:
    stockfish_path = "/Users/jetienne/Downloads/stockfish/stockfish-macos-m1-apple-silicon"  # Update this path to your Stockfish binary

    transport, engine = await chess.engine.popen_uci(stockfish_path)

    board = chess.Board()
    while not board.is_game_over():
        # Get the best move from Stockfish
        result = await engine.play(board, chess.engine.Limit(time=0.1))
        move = typing.cast(chess.Move, result.move)

        # display current board and stockfish move
        print(board.unicode())
        print(move.uci())

        # Play the move on the board
        board.push(move)

        # evaluate the resulting position and display it
        stockfish = Stockfish(path=stockfish_path)
        stockfish.set_fen_position(board.fen())
        evaluation = stockfish.get_evaluation()
        print(f"Stockfish evaluation: {evaluation}")

    await engine.quit()


asyncio.run(main())
