from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random
import numpy as np

EMPTY = 0
BLACK = 1
WHITE = -1

@dataclass
class StepResult:
    board: np.ndarray
    next_player: int
    done: bool
    info: dict

class VerticalGo:
    def __init__(self, cols: int = 7, rows: int = 6, komi: float = 0.0, target_points: Optional[int] = None):
        self.cols = cols
        self.rows = rows
        self.komi = komi
        self.target_points = target_points
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)  # row 0 bottom
        self.to_play = BLACK
        self.captures = {BLACK: 0, WHITE: 0}
        self.passes_in_row = 0
        self.last_board = None
        self.history = []
        return self.board.copy()

    def copy(self) -> "VerticalGo":
        g = VerticalGo(self.cols, self.rows, self.komi, self.target_points)
        g.board = self.board.copy()
        g.to_play = int(self.to_play)
        g.captures = dict(self.captures)
        g.passes_in_row = int(self.passes_in_row)
        g.last_board = None if self.last_board is None else self.last_board.copy()
        g.history = [b.copy() for b in self.history]
        return g

    def _in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _neighbors(self, r, c):
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            rr, cc = r+dr, c+dc
            if self._in_bounds(rr, cc):
                yield rr, cc

    def _find_group(self, board: np.ndarray, start: Tuple[int,int]):
        sr, sc = start
        color = int(board[sr, sc])
        assert color != EMPTY
        stack = [(sr, sc)]
        seen = {(sr, sc)}
        group = []
        liberties = set()
        while stack:
            r, c = stack.pop()
            group.append((r, c))
            for rr, cc in self._neighbors(r, c):
                v = int(board[rr, cc])
                if v == EMPTY:
                    liberties.add((rr, cc))
                elif v == color and (rr, cc) not in seen:
                    seen.add((rr, cc))
                    stack.append((rr, cc))
        return group, liberties

    def _settle(self, board: np.ndarray):
        for c in range(self.cols):
            col = board[:, c]
            stones = col[col != EMPTY]
            new_col = np.zeros_like(col)
            new_col[:len(stones)] = stones
            board[:, c] = new_col
        return board

    def _drop_row(self, board: np.ndarray, col: int) -> Optional[int]:
        for r in range(self.rows):
            if int(board[r, col]) == EMPTY:
                return r
        return None

    def is_full(self):
        return bool(np.all(self.board[self.rows-1, :] != EMPTY))

    def _apply_captures_inplace(self, board: np.ndarray, placed: Tuple[int,int], player: int) -> int:
        opp = -player
        captured = 0
        pr, pc = placed
        to_check = set()
        for rr, cc in self._neighbors(pr, pc):
            if int(board[rr, cc]) == opp:
                to_check.add((rr, cc))

        visited = set()
        for s in list(to_check):
            if s in visited:
                continue
            group, libs = self._find_group(board, s)
            for p in group:
                visited.add(p)
            if len(libs) == 0:
                for (r, c) in group:
                    board[r, c] = EMPTY
                captured += len(group)
        return captured

    def is_legal(self, action: int) -> bool:
        if action == self.cols:  # pass
            return True
        if not (0 <= action < self.cols):
            return False
        r = self._drop_row(self.board, action)
        if r is None:
            return False

        b = self.board.copy()
        b[r, action] = self.to_play
        captured = self._apply_captures_inplace(b, placed=(r, action), player=self.to_play)
        b = self._settle(b)

        # suicide: if no captures, cannot create any 0-liberty group of mover
        if captured == 0:
            seen = set()
            for rr in range(self.rows):
                for cc in range(self.cols):
                    if int(b[rr, cc]) == self.to_play and (rr, cc) not in seen:
                        group, libs = self._find_group(b, (rr, cc))
                        for p in group:
                            seen.add(p)
                        if len(libs) == 0:
                            return False

        # simple ko
        if self.last_board is not None and np.array_equal(b, self.last_board):
            return False
        return True

    def legal_moves(self) -> List[int]:
        moves = []
        for c in range(self.cols):
            if self._drop_row(self.board, c) is not None and self.is_legal(c):
                moves.append(c)
        moves.append(self.cols)  # pass
        return moves

    def step(self, action: int) -> StepResult:
        if action == self.cols:
            self.last_board = self.board.copy()
            self.history.append(self.board.copy())
            self.passes_in_row += 1
            done = (self.passes_in_row >= 2) or self.is_full()
            self.to_play *= -1
            return StepResult(self.board.copy(), self.to_play, done, {'pass': True})

        if not self.is_legal(action):
            raise ValueError(f"Illegal move: {action}")

        self.passes_in_row = 0
        self.last_board = self.board.copy()

        r = self._drop_row(self.board, action)
        assert r is not None
        self.board[r, action] = self.to_play
        captured = self._apply_captures_inplace(self.board, placed=(r, action), player=self.to_play)
        if captured > 0:
            self.captures[self.to_play] += captured
        self.board = self._settle(self.board)

        self.history.append(self.board.copy())
        done = self.is_full()
        info = {'captured': captured, 'pass': False}
        self.to_play *= -1
        return StepResult(self.board.copy(), self.to_play, done, info)

    def score(self) -> Dict[int, float]:
        black_stones = int(np.sum(self.board == BLACK))
        white_stones = int(np.sum(self.board == WHITE))
        black = black_stones + int(self.captures[BLACK])
        white = white_stones + int(self.captures[WHITE]) + float(self.komi)
        return {BLACK: float(black), WHITE: float(white)}

    def winner(self) -> Optional[int]:
        # Target-score win condition (optional)
        if self.target_points is not None:
            sc_now = self.score()
            if sc_now[BLACK] >= float(self.target_points) and sc_now[WHITE] >= float(self.target_points):
                # If both cross the line on the same move, fall back to higher score.
                if abs(sc_now[BLACK] - sc_now[WHITE]) < 1e-9:
                    return 0
                return BLACK if sc_now[BLACK] > sc_now[WHITE] else WHITE
            if sc_now[BLACK] >= float(self.target_points):
                return BLACK
            if sc_now[WHITE] >= float(self.target_points):
                return WHITE

        if not (self.is_full() or self.passes_in_row >= 2):
            return None
        sc = self.score()
        if abs(sc[BLACK] - sc[WHITE]) < 1e-9:
            return 0
        return BLACK if sc[BLACK] > sc[WHITE] else WHITE


def _center_bias(col: int, cols: int = 7) -> float:
    center = (cols - 1) / 2.0
    return -abs(col - center)


def _mobility(env: VerticalGo, player: int) -> int:
    e = env.copy()
    e.to_play = player
    return len(e.legal_moves())


def evaluate(env: VerticalGo, player: int) -> float:
    sc = env.score()
    return (sc[player] - sc[-player]) + 0.05 * _mobility(env, player)


class RandomBot:
    name = "Random"
    def select(self, env: VerticalGo) -> int:
        return random.choice(env.legal_moves())


class CaptureBot:
    name = "Greedy Capture"
    def select(self, env: VerticalGo) -> int:
        best = None
        best_key = (-1e18, -1e18)
        for a in env.legal_moves():
            e = env.copy()
            before = e.captures[e.to_play]
            e.step(a)
            moved = -e.to_play
            gained = e.captures[moved] - before

            # opponent best immediate capture (we want to minimize this)
            opp = e.to_play
            worst = 0
            for oa in e.legal_moves():
                ee = e.copy()
                b2 = ee.captures[opp]
                ee.step(oa)
                moved2 = -ee.to_play
                opp_gain = ee.captures[moved2] - b2
                worst = max(worst, opp_gain)

            key = (gained - 0.7*worst, _center_bias(a, env.cols))
            if key > best_key:
                best_key = key
                best = a
        return int(best if best is not None else env.cols)


class LookaheadBot:
    name = "Lookahead"
    def __init__(self, depth: int = 3):
        self.depth = int(depth)

    def select(self, env: VerticalGo) -> int:
        player = env.to_play

        def minimax(e: VerticalGo, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
            w = e.winner()
            if w is not None:
                if w == 0:
                    return 0.0
                return 999.0 if w == player else -999.0
            if depth == 0:
                return evaluate(e, player)

            moves = e.legal_moves()
            moves.sort(key=lambda a: _center_bias(a, e.cols), reverse=True)

            if maximizing:
                val = -1e18
                for a in moves:
                    ee = e.copy(); ee.step(a)
                    val = max(val, minimax(ee, depth-1, alpha, beta, False))
                    alpha = max(alpha, val)
                    if beta <= alpha:
                        break
                return val
            else:
                val = 1e18
                for a in moves:
                    ee = e.copy(); ee.step(a)
                    val = min(val, minimax(ee, depth-1, alpha, beta, True))
                    beta = min(beta, val)
                    if beta <= alpha:
                        break
                return val

        best_a = None
        best_v = -1e18
        moves = env.legal_moves()
        moves.sort(key=lambda a: _center_bias(a, env.cols), reverse=True)
        for a in moves:
            e2 = env.copy(); e2.step(a)
            v = minimax(e2, self.depth-1, -1e18, 1e18, False)
            if v > best_v:
                best_v = v
                best_a = a
        return int(best_a if best_a is not None else env.cols)


def get_bot(name: str):
    if name == "Random":
        return RandomBot()
    if name == "Greedy Capture":
        return CaptureBot()
    if name == "Lookahead (Depth 3)":
        return LookaheadBot(depth=3)
    if name == "Lookahead (Depth 4)":
        return LookaheadBot(depth=4)
    return CaptureBot()
