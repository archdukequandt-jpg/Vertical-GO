from __future__ import annotations
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Circle

from vertical_go import VerticalGo, get_bot, BLACK, WHITE, EMPTY

st.set_page_config(page_title="Vertical GO", page_icon="🟦", layout="wide")

BOARD_PRESETS = {
    "7 x 6 (Classic)": (7, 6),
    "9 x 7 (Medium)": (9, 7),
    "11 x 8 (Large)": (11, 8),
}

GAME_MODES = {
    "First to 50 points": 50,
    "First to 100 points": 100,
}


def draw_board(board: np.ndarray, falling=None):
    rows, cols = board.shape
    fig, ax = plt.subplots(figsize=(cols*0.9, rows*0.9))
    ax.set_xlim(-0.5, cols-0.5)
    ax.set_ylim(-0.5, rows-0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row (0 = bottom)")

    for r in range(rows):
        for c in range(cols):
            v = int(board[r, c])
            if v == EMPTY:
                continue
            face = 'C0' if v == BLACK else 'C1'
            ax.add_patch(Circle((c, r), 0.42, facecolor=face, edgecolor='black', linewidth=1.0))
    # Optional falling piece animation: falling=(col, y, color)
    if falling is not None:
        fcol, fy, fcolor = falling
        face = 'C0' if int(fcolor) == BLACK else 'C1'
        ax.add_patch(Circle((fcol, fy), 0.42, facecolor=face, edgecolor='black', linewidth=1.0, alpha=0.95))

    return fig


def animate_drop(board_ph, board: np.ndarray, col: int, target_row: int, color: int, max_duration_s: float):
    """Animate a falling stone in the *main board area* using the provided placeholder."""
    rows, _ = board.shape
    y_start = rows - 0.2
    y_end = float(target_row)
    # Total animation duration capped to 1s (usually much faster)
    total = float(max_duration_s)
    if total <= 0:
        total = 0.25
    total = min(total, 0.40)
    steps = max(12, int(abs(y_start - y_end) * 18))
    sleep_dt = total / steps
    for i in range(steps + 1):
        t = i / steps
        y = y_start + (y_end - y_start) * t
        fig = draw_board(board, falling=(col, y, color))
        board_ph.pyplot(fig, clear_figure=True)
        time.sleep(sleep_dt)



def ensure_state():
    if 'board_preset' not in st.session_state:
        st.session_state.board_preset = "7 x 6 (Classic)"
    if 'game_mode' not in st.session_state:
        st.session_state.game_mode = "First to 50 points"

    cols, rows = BOARD_PRESETS.get(st.session_state.board_preset, (7, 6))
    target_points = GAME_MODES.get(st.session_state.game_mode, 50)

    if 'env' not in st.session_state:
        st.session_state.env = VerticalGo(cols=cols, rows=rows, target_points=target_points)
    else:
        # Keep env in sync with selected mode (board size changes require a new game).
        st.session_state.env.target_points = target_points

    if 'human_color' not in st.session_state:
        st.session_state.human_color = BLACK
    if 'last_msg' not in st.session_state:
        st.session_state.last_msg = ""
    if 'pending_ai' not in st.session_state:
        st.session_state.pending_ai = False
    if 'bot_name' not in st.session_state:
        st.session_state.bot_name = "Greedy Capture"
    if 'bot' not in st.session_state:
        st.session_state.bot = get_bot(st.session_state.bot_name)



def status_text(env: VerticalGo, human_color: int):
    sc = env.score()
    w = env.winner()
    turn = "You (Black ●)" if env.to_play == human_color else "AI (White ○)"
    end = ""
    if w is not None:
        if w == 0:
            end = "Game over: Draw."
        elif w == human_color:
            end = "Game over: You win."
        else:
            end = "Game over: AI wins."
    return turn, sc[human_color], sc[-human_color], end


def ai_turn(board_ph=None):
    env: VerticalGo = st.session_state.env
    human_color: int = st.session_state.human_color

    if board_ph is None:
        board_ph = st.empty()
    bot = st.session_state.bot

    if board_ph is None:
        board_ph = st.empty()

    if env.winner() is not None:
        return
    if env.to_play != -human_color:
        return

    a = bot.select(env)
    if a != env.cols and animate and board_ph is not None:
        r = env._drop_row(env.board, a)  # type: ignore[attr-defined]
        if r is not None:
            animate_drop(board_ph, env.board.copy(), a, r, env.to_play, anim_speed)
    res = env.step(a)
    st.session_state.last_msg = f"AI played: {'Pass' if a==env.cols else f'Col {a}'} (captured {res.info.get('captured',0)})"


def human_move(action: int, board_ph=None):
    env: VerticalGo = st.session_state.env
    human_color: int = st.session_state.human_color

    if board_ph is None:
        board_ph = st.empty()
    if env.winner() is not None:
        st.session_state.last_msg = "Game is over. Start a new game."
        return
    if env.to_play != human_color:
        st.session_state.last_msg = "Not your turn."
        return
    try:
        if action != env.cols and animate:
            r = env._drop_row(env.board, action)  # type: ignore[attr-defined]
            if r is not None:
                animate_drop(board_ph, env.board.copy(), action, r, env.to_play, anim_speed)
        res = env.step(action)
        st.session_state.last_msg = f"You played: {'Pass' if action==env.cols else f'Col {action}'} (captured {res.info.get('captured',0)})"
    except Exception as e:
        st.session_state.last_msg = f"Illegal move: {e}"
        return

    ai_turn(board_ph)


def new_game(human_color: int):
    cols, rows = BOARD_PRESETS.get(st.session_state.board_preset, (7, 6))
    target_points = GAME_MODES.get(st.session_state.game_mode, 50)
    komi = float(getattr(st.session_state.env, "komi", 0.0)) if "env" in st.session_state else 0.0

    st.session_state.env = VerticalGo(cols=cols, rows=rows, komi=komi, target_points=target_points)
    st.session_state.human_color = human_color
    st.session_state.last_msg = "New game started."
    if st.session_state.env.to_play != st.session_state.human_color:
        st.session_state.pending_ai = True



ensure_state()

st.title('Vertical GO')
st.caption('Designed & Created by Ryan Childs')


# Sidebar
st.sidebar.title("Vertical GO")
st.sidebar.caption("Prebuilt AI opponents — no training required.")
st.sidebar.markdown("**Designed & Created by Ryan Childs**")
animate = st.sidebar.checkbox("Animate drops", value=True)
anim_speed = st.sidebar.slider("Drop animation duration (seconds)", min_value=0.03, max_value=0.40, value=0.12, step=0.01)


bot_name = st.sidebar.selectbox(
    "Choose AI opponent",
    ["Random", "Greedy Capture", "Lookahead (Depth 3)", "Lookahead (Depth 4)"],
    index=["Random", "Greedy Capture", "Lookahead (Depth 3)", "Lookahead (Depth 4)"].index(st.session_state.bot_name),
)
if bot_name != st.session_state.bot_name:
    st.session_state.bot_name = bot_name
    st.session_state.bot = get_bot(bot_name)

komi = st.sidebar.number_input("Komi (optional)", value=float(st.session_state.env.komi), step=0.5)
st.session_state.env.komi = float(komi)

# Board / mode settings
board_preset = st.sidebar.selectbox(
    "Board size",
    list(BOARD_PRESETS.keys()),
    index=list(BOARD_PRESETS.keys()).index(st.session_state.board_preset),
)
if board_preset != st.session_state.board_preset:
    st.session_state.board_preset = board_preset
    # Changing board size requires a fresh game
    new_game(st.session_state.human_color)

game_mode = st.sidebar.radio(
    "Game mode",
    list(GAME_MODES.keys()),
    index=list(GAME_MODES.keys()).index(st.session_state.game_mode),
)
if game_mode != st.session_state.game_mode:
    st.session_state.game_mode = game_mode
    # Update target points immediately; no need to restart
    st.session_state.env.target_points = GAME_MODES[game_mode]


st.sidebar.markdown('---')
st.sidebar.subheader('New game')
human_side = st.sidebar.radio('You play as', ['Black (first)', 'White (second)'], index=0)
if st.sidebar.button('Start new game'):
    new_game(BLACK if human_side.startswith('Black') else WHITE)

# Main
st.title("🟦 Vertical GO")
st.write("Drop stones into columns; captures happen Go-style when groups have 0 liberties. Captured stones are removed and the board settles with gravity.")
st.write(f"**AI Opponent:** {st.session_state.bot_name}")

env: VerticalGo = st.session_state.env
human_color: int = st.session_state.human_color

turn, you_score, ai_score, end = status_text(env, human_color)

top1, top2, top3, top4 = st.columns([1.2, 1.2, 1.2, 2.0])
top1.metric('To play', turn)
top2.metric('Your captures', env.captures[human_color])
top3.metric('AI captures', env.captures[-human_color])
top4.metric('Score (You / AI)', f"{you_score:.1f} / {ai_score:.1f}")

if end:
    st.success(end)

st.info(st.session_state.last_msg if st.session_state.last_msg else 'Make a move.')

left, right = st.columns([2.2, 1.0], gap='large')
with left:
    board_ph = st.empty()
    board_ph.pyplot(draw_board(env.board), clear_figure=True)
    # If AI is to play immediately (e.g., you chose White), animate & execute on the main board.
    if st.session_state.get('pending_ai', False) and env.winner() is None:
        st.session_state.pending_ai = False
        ai_turn(board_ph)
        st.rerun()


with right:
    st.subheader('Your move')
    st.caption('Choose a column (0–6) or Pass. Illegal moves are rejected (suicide/ko/full column).')
    btn_cols = st.columns(4)
    for c in range(env.cols):
        with btn_cols[c % 4]:
            if st.button(f"Drop in {c}", use_container_width=True, disabled=(env.winner() is not None)):
                human_move(c, board_ph)
                st.rerun()
    if st.button('Pass', use_container_width=True, disabled=(env.winner() is not None)):
        human_move(env.cols, board_ph)
        st.rerun()

st.markdown('---')
with st.expander('Rules & notes (this implementation)'):
    st.markdown('''
1. Board is 7×6. Row 0 is bottom.
2. A move chooses a column; stone falls to lowest empty cell.
3. After placement, any adjacent opponent group with **0 liberties** is captured.
4. After captures, stones settle downward (gravity).
5. **Suicide** is illegal unless it captures opponent stones.
6. **Simple ko**: you cannot recreate the immediately previous board position.
7. Game ends on **two consecutive passes** or a full board.
Scoring: stones-on-board + captures (komi optional).
''')
