#!/usr/bin/env python3
"""
Monkeytype-style TUI typing speed game — 30-second words mode.

Usage
-----
    python misc/typing_game.py

Controls
--------
    Any printable key   Start the game (on the start screen)
    Space               Confirm a correctly-typed word and advance
    Backspace           Fix a typo (required before you can advance)
    Esc                 Quit at any time
    r                   Restart (on the results screen)

Scoring
-------
    WPM = words_completed ÷ 0.5   (30 s = 0.5 min)
    Only words whose typed text exactly matches the target word when
    Space is pressed are counted.  Advancing past a word with an
    uncorrected error is not allowed.
"""

from __future__ import annotations

import curses
import random
import time

# ── word pool ─────────────────────────────────────────────────────────────────

WORDS: list[str] = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "great", "between", "need",
    "large", "often", "hand", "high", "place", "hold", "turn", "help",
    "call", "follow", "show", "around", "form", "small", "set", "put",
    "end", "another", "big", "play", "spell", "air", "away", "animal",
    "house", "point", "page", "letter", "mother", "answer", "found", "still",
    "learn", "should", "world", "next", "never", "ask", "until", "change",
    "while", "start", "city", "earth", "eyes", "light", "thought", "head",
    "under", "story", "saw", "left", "few", "along", "might", "close",
    "something", "seem", "hard", "open", "beginning", "life", "always",
    "those", "both", "paper", "together", "got", "group", "run", "children",
    "side", "feet", "car", "mile", "night", "walk", "white", "sea", "began",
    "grow", "took", "river", "four", "carry", "state", "once", "book",
    "hear", "stop", "second", "later", "idea", "enough", "eat", "face",
    "watch", "far", "real", "almost", "let", "above", "girl", "mountain",
    "cut", "young", "talk", "soon", "list", "song", "being", "leave",
    "family", "body", "music", "color", "stand", "sun", "fish", "area",
    "mark", "dog", "horse", "birds", "problem", "complete", "room", "knew",
    "since", "ever", "piece", "told", "usually", "friends", "easy", "heard",
    "order", "red", "door", "sure", "become", "top", "ship", "across",
    "today", "during", "short", "better", "best", "however", "low", "hours",
    "black", "happened", "whole", "measure", "remember", "early", "waves",
    "reached", "listen", "wind", "rock", "space", "covered", "fast",
    "several", "himself", "toward", "five", "step", "morning", "passed",
    "true", "hundred", "against", "pattern", "table", "north", "slowly",
    "money", "map", "farm", "pulled", "draw", "voice", "power", "town",
    "fine", "drive", "warm", "common", "bring", "explain", "dry", "though",
    "language", "shape", "deep", "yes", "clear", "filled", "heat", "full",
    "hot", "check", "object", "rule", "among", "able", "size", "dark",
    "ball", "material", "special", "heavy", "pair", "circle", "include",
    "built", "matter", "square", "perhaps", "felt", "suddenly", "test",
    "direction", "ready", "divided", "general", "energy", "subject", "moon",
    "region", "return", "believe", "dance", "members", "picked", "simple",
    "cells", "paint", "mind", "love", "cause", "rain", "train", "blue",
    "wish", "drop", "window", "distance", "heart", "sum", "summer", "wall",
    "forest", "probably", "legs", "sat", "main", "winter", "wide", "written",
    "length", "reason", "kept", "interest", "arms", "brother", "race",
    "present", "beautiful", "store", "job", "edge", "past", "sign",
    "record", "finished", "wild", "happy", "gone", "sky", "glass", "million",
    "west", "lay", "weather", "root", "meet", "third", "months", "raised",
    "soft", "clothes", "flowers", "shall", "teacher", "held", "describe",
    "cross", "speak", "solve", "appear", "metal", "son", "either", "ice",
    "sleep", "village", "result", "jumped", "snow", "ride", "care", "floor",
    "hill", "pushed", "baby", "buy", "outside", "everything", "tall",
    "already", "instead", "soil", "bed", "copy", "free", "hope", "spring",
    "case", "laughed", "nation", "quite", "type", "themselves", "temperature",
    "bright", "lead", "everyone", "method", "section", "lake", "hair",
    "age", "amount", "scale", "pounds", "although", "per", "broken",
    "moment", "tiny", "possible", "gold", "milk", "quiet", "natural",
    "lot", "stone", "act", "middle", "speed", "count", "someone", "sail",
    "rolled", "bear", "wonder", "smiled", "angle", "bottom", "trip",
    "hole", "poor", "fight", "surprise", "died", "beat", "exactly",
    "remain", "dress", "iron", "fingers", "row", "least", "catch",
    "climbed", "wrote", "shouted", "continued", "else", "plains", "gas",
    "burning", "design", "joined", "foot", "law", "ears", "grass", "grew",
    "skin", "valley", "key", "brown", "trouble", "cool", "cloud", "lost",
    "sent", "wear", "bad", "save", "experiment", "engine", "alone",
    "drawing", "east", "pay", "single", "touch", "express", "mouth",
    "yard", "equal", "decimal", "water", "little", "number", "off",
    "always", "move", "try", "kind", "hand", "picture", "again", "change",
    "off", "play", "spell", "away", "animal", "house", "point", "page",
    "letter", "mother", "answer", "found", "still", "learn", "plant",
    "cover", "food", "sun", "four", "between", "state", "keep", "eye",
    "never", "last", "let", "thought", "city", "tree", "cross", "farm",
    "hard", "start", "might", "story", "saw", "far", "sea", "draw",
    "left", "late", "run", "don't", "while", "press", "close", "night",
    "real", "life", "few", "north", "open", "seem", "together", "next",
    "white", "children", "begin", "got", "walk", "example", "ease",
    "paper", "group", "always", "music", "those", "both", "mark",
    "often", "letter", "until", "mile", "river", "car", "feet", "care",
    "second", "book", "carry", "took", "science", "eat", "room", "friend",
    "began", "idea", "fish", "mountain", "stop", "once", "base", "hear",
    "horse", "cut", "sure", "watch", "color", "face", "wood", "main",
    "enough", "plain", "girl", "usual", "young", "ready", "above",
    "ever", "red", "list", "though", "feel", "talk", "bird", "soon",
    "body", "dog", "family", "direct", "pose", "leave", "song", "measure",
    "door", "product", "black", "short", "numeral", "class", "wind",
    "question", "happen", "complete", "ship", "area", "half", "rock",
    "order", "fire", "south", "problem", "piece", "told", "knew",
    "pass", "since", "top", "whole", "king", "space", "heard", "best",
    "hour", "better", "true", "during", "hundred", "five", "remember",
    "step", "early", "hold", "west", "ground", "interest", "reach",
    "fast", "verb", "sing", "listen", "six", "table", "travel", "less",
    "morning", "ten", "simple", "several", "vowel", "toward", "war",
    "lay", "against", "pattern", "slow", "center", "love", "person",
    "money", "serve", "appear", "road", "map", "rain", "rule", "govern",
    "pull", "cold", "notice", "voice", "unit", "power", "town", "fine",
    "drive", "lead", "cry", "dark", "machine", "note", "wait", "plan",
    "figure", "star", "box", "noun", "field", "rest", "able", "pound",
    "done", "beauty", "drive", "stood", "contain", "front", "teach",
    "week", "final", "gave", "green", "oh", "quick", "develop", "ocean",
    "warm", "free", "minute", "strong", "special", "mind", "behind",
    "clear", "tail", "produce", "fact", "street", "inch", "multiply",
    "nothing", "course", "stay", "wheel", "full", "force", "blue",
    "object", "decide", "surface", "deep", "moon", "island", "foot",
    "system", "busy", "test", "record", "boat", "common", "gold",
    "possible", "plane", "steady", "dry", "wonder", "laugh", "thousand",
    "ago", "ran", "check", "game", "shape", "equate", "hot", "miss",
    "brought", "heat", "snow", "tire", "bring", "yes", "distant",
    "fill", "east", "paint", "language", "among", "grand", "ball",
    "yet", "wave", "drop", "heart", "am", "present", "heavy", "dance",
    "engine", "position", "arm", "wide", "sail", "material", "size",
    "vary", "settle", "speak", "weight", "general", "ice", "matter",
    "circle", "pair", "include", "divide", "syllable", "felt", "perhaps",
    "pick", "sudden", "count", "square", "reason", "length", "represent",
    "art", "subject", "region", "energy", "hunt", "probable", "bed",
    "brother", "egg", "ride", "cell", "believe", "fraction", "forest",
    "sit", "race", "window", "store", "summer", "train", "sleep",
    "prove", "lone", "leg", "exercise", "wall", "catch", "mount",
    "wish", "sky", "board", "joy", "winter", "sat", "written", "wild",
    "instrument", "kept", "glass", "grass", "cow", "job", "edge",
    "sign", "visit", "past", "soft", "fun", "bright", "gas", "weather",
    "month", "million", "bear", "finish", "happy", "hope", "flower",
    "clothe", "strange", "gone", "jump", "baby", "eight", "village",
    "meet", "root", "buy", "raise", "solve", "metal", "whether",
    "push", "seven", "paragraph", "third", "shall", "held", "hair",
    "describe", "cook", "floor", "either", "result", "burn", "hill",
    "safe", "cat", "century", "consider", "type", "law", "bit",
    "coast", "copy", "phrase", "silent", "tall", "sand", "soil",
    "roll", "temperature", "finger", "industry", "value", "fight",
    "lie", "beat", "excite", "natural", "view", "sense", "ear",
    "else", "quite", "broke", "case", "middle", "kill", "son",
    "lake", "moment", "scale", "loud", "spring", "observe", "child",
    "straight", "consonant", "nation", "dictionary", "milk", "speed",
    "method", "organ", "pay", "age", "section", "dress", "cloud",
    "surprise", "quiet", "stone", "tiny", "climb", "cool", "design",
    "poor", "lot", "experiment", "bottom", "key", "iron", "single",
    "stick", "flat", "twenty", "skin", "smile", "crease", "hole",
    "trade", "melody", "trip", "office", "receive", "row", "mouth",
    "exact", "symbol", "die", "least", "trouble", "shout", "except",
    "wrote", "seed", "tone", "join", "suggest", "clean", "break",
    "lady", "yard", "rise", "bad", "blow", "oil", "blood", "touch",
    "grew", "cent", "mix", "team", "wire", "cost", "lost", "brown",
    "wear", "garden", "equal", "sent", "choose", "fell", "fit",
    "flow", "fair", "bank", "collect", "save", "control", "decimal",
]

DURATION = 30  # seconds

# ── color pair IDs ────────────────────────────────────────────────────────────

C_DEFAULT = 1   # terminal default        — untyped characters
C_CORRECT = 2   # green                  — correct chars / completed words
C_ERROR   = 3   # red                    — typo
C_CURSOR  = 4   # yellow + underline     — next char to type
C_CHROME  = 5   # cyan                   — UI borders and labels
C_DIM     = 6   # dim                    — upcoming words


# ── helpers ───────────────────────────────────────────────────────────────────

def pick_words(n: int = 120) -> list[str]:
    return random.choices(WORDS, k=n)


def layout_words(
    words: list[str], width: int
) -> list[list[tuple[int, str]]]:
    """Wrap *words* into rows of at most *width* characters.

    Returns a list of rows; each row is a list of ``(word_index, word)``
    tuples.
    """
    rows: list[list[tuple[int, str]]] = []
    row: list[tuple[int, str]] = []
    used = 0

    for i, word in enumerate(words):
        # A leading space is needed for every word that is not the first on
        # its row.
        need = len(word) + (1 if row else 0)
        if row and used + need > width:
            rows.append(row)
            row = [(i, word)]
            used = len(word)
        else:
            row.append((i, word))
            used += need

    if row:
        rows.append(row)

    return rows


def row_of(
    layout: list[list[tuple[int, str]]], target: int
) -> int:
    """Return the row index in *layout* that contains word *target*."""
    for r, row in enumerate(layout):
        if any(idx == target for idx, _ in row):
            return r
    return 0


# ── drawing ───────────────────────────────────────────────────────────────────

def safe_str(scr, y: int, x: int, text: str, attr: int = 0) -> None:
    """addstr that silently swallows out-of-bounds errors."""
    try:
        scr.addstr(y, x, text, attr)
    except curses.error:
        pass


def hline(scr, y: int, w: int) -> None:
    safe_str(scr, y, 0, "─" * w, curses.color_pair(C_CHROME))


def draw_words(
    scr,
    layout: list[list[tuple[int, str]]],
    current_idx: int,
    typed: str,
    y_start: int,
    y_end: int,        # exclusive
    x_off: int,
    scroll_row: int,   # first layout row to display
) -> None:
    """Render the word area between rows *y_start* and *y_end*."""
    for rel in range(y_end - y_start):
        row_idx = scroll_row + rel
        if row_idx >= len(layout):
            break
        row = layout[row_idx]
        x = x_off
        y = y_start + rel

        for w_idx, word in row:
            if w_idx < current_idx:
                # already completed — whole word green
                safe_str(scr, y, x, word, curses.color_pair(C_CORRECT))
            elif w_idx == current_idx:
                # active word — character-by-character colouring
                for j, ch in enumerate(word):
                    if j < len(typed):
                        attr = (
                            curses.color_pair(C_CORRECT)
                            if typed[j] == ch
                            else curses.color_pair(C_ERROR)
                        )
                    elif j == len(typed):
                        attr = curses.color_pair(C_CURSOR) | curses.A_UNDERLINE
                    else:
                        attr = curses.color_pair(C_DEFAULT)
                    safe_str(scr, y, x + j, ch, attr)
            else:
                # upcoming — dim
                safe_str(scr, y, x, word, curses.color_pair(C_DIM) | curses.A_DIM)

            x += len(word) + 1  # advance past the word + one space


# ── main loop ─────────────────────────────────────────────────────────────────

def run(scr) -> None:  # noqa: C901  (acceptable length for a game loop)
    curses.curs_set(0)
    scr.nodelay(True)
    scr.timeout(50)

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_DEFAULT, -1,                  -1)
    curses.init_pair(C_CORRECT, curses.COLOR_GREEN,  -1)
    curses.init_pair(C_ERROR,   curses.COLOR_RED,    -1)
    curses.init_pair(C_CURSOR,  curses.COLOR_YELLOW, -1)
    curses.init_pair(C_CHROME,  curses.COLOR_CYAN,   -1)
    curses.init_pair(C_DIM,     -1,                  -1)

    WAIT, PLAY, DONE = 0, 1, 2

    # mutable game state (reset via closure below)
    words: list[str] = []
    layout: list[list[tuple[int, str]]] = []
    cur: int = 0
    typed: str = ""
    has_error: bool = False
    start_time: float = 0.0
    completed: int = 0

    def reset() -> None:
        nonlocal words, layout, cur, typed, has_error, start_time, completed
        words      = pick_words(120)
        cur        = 0
        typed      = ""
        has_error  = False
        start_time = 0.0
        completed  = 0

    reset()
    state = WAIT

    # ── game loop ─────────────────────────────────────────────────────────────
    while True:
        h, w = scr.getmaxyx()
        scr.erase()

        # Recompute layout every frame so resize is handled gracefully.
        word_area_w = max(10, w - 4)
        layout = layout_words(words, word_area_w)

        # ── WAIT ──────────────────────────────────────────────────────────────
        if state == WAIT:
            title = "  TYPING SPEED TEST  "
            safe_str(scr, 1, (w - len(title)) // 2, title,
                     curses.color_pair(C_CHROME) | curses.A_BOLD | curses.A_REVERSE)

            info_lines = [
                f"You have {DURATION} seconds to type as many words as you can.",
                "Typos must be corrected with Backspace before you can advance.",
                "Press Space after each word to confirm it.  Only exact matches count.",
            ]
            for i, line in enumerate(info_lines):
                safe_str(scr, 3 + i, max(0, (w - len(line)) // 2), line,
                         curses.color_pair(C_DEFAULT))

            # word preview (first 3 layout rows)
            preview_y0 = 7
            preview_y1 = min(h - 3, preview_y0 + 3)
            if preview_y1 > preview_y0:
                draw_words(scr, layout, current_idx=-1, typed="",
                           y_start=preview_y0, y_end=preview_y1,
                           x_off=2, scroll_row=0)

            prompt = "── press any key to start ──"
            safe_str(scr, h - 2, max(0, (w - len(prompt)) // 2), prompt,
                     curses.color_pair(C_CHROME) | curses.A_BOLD)

            scr.refresh()
            key = scr.getch()

            if key == 27:
                return
            if key == curses.KEY_RESIZE:
                continue
            if 32 <= key <= 126:                   # any printable key → start
                state      = PLAY
                start_time = time.monotonic()
                ch = chr(key)
                if ch != ' ':
                    typed += ch
                    if typed[0] != words[cur][0]:
                        has_error = True

        # ── PLAY ──────────────────────────────────────────────────────────────
        elif state == PLAY:
            now       = time.monotonic()
            elapsed   = now - start_time
            remaining = max(0.0, DURATION - elapsed)

            if remaining == 0.0:
                state = DONE
                continue

            # ── status bar ──
            timer_text = f" {remaining:.1f}s "
            wpm_val    = int(completed / (elapsed / 60)) if elapsed >= 1.0 else 0
            wpm_text   = f" WPM: {wpm_val} "
            words_text = f" Words: {completed} "

            timer_attr = curses.color_pair(C_ERROR if remaining <= 10 else C_CHROME) | curses.A_BOLD
            safe_str(scr, 0, 1, timer_text, timer_attr)
            safe_str(scr, 0, (w - len(wpm_text)) // 2, wpm_text,
                     curses.color_pair(C_CORRECT) | curses.A_BOLD)
            safe_str(scr, 0, w - len(words_text) - 1, words_text,
                     curses.color_pair(C_DEFAULT))

            hline(scr, 1, w)

            # ── word area ──
            WORD_Y0 = 2
            WORD_Y1 = max(WORD_Y0 + 1, h - 4)

            cur_row    = row_of(layout, cur)
            scroll_row = max(0, cur_row - 1)   # one row of context above current

            draw_words(scr, layout, cur, typed,
                       y_start=WORD_Y0, y_end=WORD_Y1,
                       x_off=2, scroll_row=scroll_row)

            # ── input area ──
            hline(scr, h - 4, w)

            prefix = "  > "
            safe_str(scr, h - 3, 0, prefix, curses.color_pair(C_CHROME))

            if has_error:
                safe_str(scr, h - 3, len(prefix), typed,
                         curses.color_pair(C_ERROR) | curses.A_BOLD)
                safe_str(scr, h - 2, 2,
                         "Backspace to fix typo  |  Esc to quit",
                         curses.color_pair(C_ERROR))
            else:
                safe_str(scr, h - 3, len(prefix), typed,
                         curses.color_pair(C_CORRECT))
                safe_str(scr, h - 2, 2,
                         "Space to confirm word  |  Esc to quit",
                         curses.color_pair(C_CHROME) | curses.A_DIM)

            scr.refresh()

            # ── input handling ──
            key = scr.getch()

            if key == -1 or key == curses.KEY_RESIZE:
                continue
            if key == 27:                                  # Esc → quit
                return

            if key in (curses.KEY_BACKSPACE, 127, 8):     # Backspace
                if typed:
                    typed = typed[:-1]
                # recompute error state from scratch
                has_error = any(
                    j >= len(words[cur]) or typed[j] != words[cur][j]
                    for j in range(len(typed))
                )

            elif key == ord(' '):                          # Space → confirm
                if not has_error and typed == words[cur]:
                    completed += 1
                    cur       += 1
                    typed      = ""
                    has_error  = False
                    # replenish pool before it runs dry
                    if cur >= len(words) - 30:
                        words.extend(pick_words(40))
                # If there is an error or the word is incomplete, ignore Space.

            elif 32 < key <= 126:                         # printable (non-space)
                ch     = chr(key)
                typed += ch
                pos    = len(typed) - 1
                if pos >= len(words[cur]) or typed[pos] != words[cur][pos]:
                    has_error = True

        # ── DONE ──────────────────────────────────────────────────────────────
        elif state == DONE:
            final_wpm = int(completed / (DURATION / 60))

            title = "  TIME'S UP!  "
            safe_str(scr, 1, (w - len(title)) // 2, title,
                     curses.color_pair(C_CHROME) | curses.A_BOLD | curses.A_REVERSE)

            # large WPM display
            wpm_str   = str(final_wpm)
            wpm_label = "words per minute"
            safe_str(scr, 3, (w - len(wpm_str)) // 2, wpm_str,
                     curses.color_pair(C_CORRECT) | curses.A_BOLD)
            safe_str(scr, 4, (w - len(wpm_label)) // 2, wpm_label,
                     curses.color_pair(C_CHROME))

            # stats table
            stats = [
                ("Words completed", str(completed)),
                ("Duration",        f"{DURATION}s"),
                ("WPM",             str(final_wpm)),
            ]
            col = max(0, (w - 28) // 2)
            for i, (k, v) in enumerate(stats):
                safe_str(scr, 6 + i, col, f"{k:<20}{v}",
                         curses.color_pair(C_DEFAULT))

            # rating
            if final_wpm >= 80:
                rating, rc = "Outstanding!", C_CHROME
            elif final_wpm >= 60:
                rating, rc = "Great speed!", C_CORRECT
            elif final_wpm >= 40:
                rating, rc = "Above average — keep it up!", C_CORRECT
            elif final_wpm >= 20:
                rating, rc = "Keep practicing!", C_CURSOR
            else:
                rating, rc = "Just getting started — don't give up!", C_ERROR

            safe_str(scr, 10, max(0, (w - len(rating)) // 2), rating,
                     curses.color_pair(rc) | curses.A_BOLD)

            opts = "  [r] play again    [q] quit  "
            safe_str(scr, 12, max(0, (w - len(opts)) // 2), opts,
                     curses.color_pair(C_DEFAULT))

            scr.refresh()
            key = scr.getch()

            if key in (ord('q'), 27):
                return
            if key == ord('r'):
                reset()
                state = WAIT


def main() -> None:
    curses.wrapper(run)


if __name__ == "__main__":
    main()
