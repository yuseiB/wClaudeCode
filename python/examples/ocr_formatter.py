#!/usr/bin/env python3
"""OCR Text Formatter — WSL/Windows TUI tool.

Reads raw OCR text from the clipboard, fixes common formatting issues,
and writes the cleaned result back to the clipboard.

Usage
-----
    python python/examples/ocr_formatter.py          # read from clipboard
    python python/examples/ocr_formatter.py --stdin  # read from stdin
    python python/examples/ocr_formatter.py --math   # auto-run Claude math fix
    python python/examples/ocr_formatter.py --help

Keybindings
-----------
    y / Enter   Copy formatted text to clipboard and exit
    q / Esc     Quit without copying
    m           Run Claude math fix (requires ANTHROPIC_API_KEY)
    e           Open in $EDITOR, reload on save
    ↑ / ↓       Scroll both panes one line
    PgUp/PgDn   Scroll by page
    Tab         Switch active (highlighted) pane
"""

from __future__ import annotations

import argparse
import curses
import os
import re
import subprocess
import sys
import tempfile
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Clipboard I/O
# ---------------------------------------------------------------------------

def _is_wsl() -> bool:
    """Return True when running inside Windows Subsystem for Linux."""
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def read_clipboard() -> str:
    """Return the current clipboard text (WSL-aware)."""
    if _is_wsl():
        result = subprocess.run(
            ["powershell.exe", "-command", "Get-Clipboard"],
            capture_output=True, text=True, check=False,
        )
        return result.stdout.rstrip("\r\n")

    # Native Linux fallbacks
    for cmd in (["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"]):
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout
    raise RuntimeError("No clipboard tool found. Install xclip or xsel.")


def write_clipboard(text: str) -> None:
    """Write *text* to the system clipboard (WSL-aware)."""
    if _is_wsl():
        proc = subprocess.Popen(["clip.exe"], stdin=subprocess.PIPE)
        # clip.exe wants UTF-16 LE without BOM for best compatibility
        proc.communicate(input=text.encode("utf-16-le"))
        return

    # Native Linux fallbacks
    for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        proc.communicate(input=text.encode())
        if proc.returncode == 0:
            return
    raise RuntimeError("No clipboard tool found. Install xclip or xsel.")


# ---------------------------------------------------------------------------
# Formatting pipeline
# ---------------------------------------------------------------------------

# Sentence-ending punctuation: a line ending with these is a paragraph end.
_SENTENCE_END = re.compile(r'[.?!:…。！？]\s*$')

# Hyphenated line break: "word-\nword" → "wordword"
_SOFT_HYPHEN = re.compile(r'(\w)-\n(\w)')

# URL pattern (greedy, stops at whitespace or common delimiters)
_URL_RAW = re.compile(
    r'(?<!\{)'                         # not already inside \url{
    r'(https?://[^\s<>"{}|\\^`\[\]]+' # http/https URLs
    r'|www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s<>"{}|\\^`\[\]]*)'
)

# Already-wrapped URLs: don't double-wrap
_URL_WRAPPED = re.compile(r'\\url\{[^}]+\}')

# En-dash / minus lookalike used by some OCR engines
_EN_DASH = re.compile(r'(?<!\s)−(?!\s)')   # isolated en-dash → hyphen-minus
_MULTI_SPACE = re.compile(r'  +')           # two or more spaces → one


def _join_paragraph(lines: list[str]) -> str:
    """Join a list of lines belonging to a single paragraph.

    Lines ending with sentence-ending punctuation start a new logical sentence
    but still stay in the same paragraph (they are joined with a space).
    Lines that look like they were cut mid-sentence are simply joined.
    """
    if not lines:
        return ""
    joined: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            continue
        if joined:
            joined.append(" " + stripped)
        else:
            joined.append(stripped)
    return "".join(joined)


def _wrap_urls(text: str) -> str:
    """Replace bare URLs with ``\\url{...}`` unless already wrapped."""
    # Collect already-wrapped spans so we don't touch them
    protected: list[tuple[int, int]] = [m.span() for m in _URL_WRAPPED.finditer(text)]

    def _in_protected(start: int, end: int) -> bool:
        return any(ps <= start and end <= pe for ps, pe in protected)

    result = []
    prev = 0
    for m in _URL_RAW.finditer(text):
        s, e = m.span()
        if _in_protected(s, e):
            result.append(text[prev:e])
            prev = e
            continue
        result.append(text[prev:s])
        result.append(r"\url{" + m.group() + "}")
        prev = e
    result.append(text[prev:])
    return "".join(result)


def _math_heuristics(text: str) -> str:
    """Apply conservative, low-false-positive math OCR corrections."""
    # En-dash used as minus sign in math: "x − y" is fine; lone "−" → "-"
    text = _EN_DASH.sub("-", text)
    # Collapse accidental double spaces (common around operators)
    text = _MULTI_SPACE.sub(" ", text)
    return text


def format_ocr(text: str) -> str:
    """Full OCR formatting pipeline.

    Steps
    -----
    1. Soft-hyphen removal:  ``for-\\nmatted`` → ``formatted``
    2. Paragraph-aware line joining (blank lines are paragraph separators)
    3. URL wrapping: bare URLs → ``\\url{...}``
    4. Conservative math heuristics
    """
    # Step 1: remove soft hyphens before splitting into paragraphs
    text = _SOFT_HYPHEN.sub(r'\1\2', text)

    # Step 2: paragraph-aware line joining
    # Split into paragraphs at blank lines (one or more)
    raw_paragraphs = re.split(r'\n{2,}', text)
    joined_paragraphs: list[str] = []
    for para in raw_paragraphs:
        lines = para.split('\n')
        joined_paragraphs.append(_join_paragraph(lines))

    text = "\n\n".join(p for p in joined_paragraphs if p)

    # Step 3: URL wrapping
    text = _wrap_urls(text)

    # Step 4: math heuristics
    text = _math_heuristics(text)

    return text


# ---------------------------------------------------------------------------
# Claude math fix
# ---------------------------------------------------------------------------

_MATH_PROMPT = """\
The following text is an OCR result from a scientific or mathematical textbook.
Your task: fix LaTeX math formatting ONLY. Do not alter any non-math text.
- Wrap inline math expressions in $...$
- Wrap display (standalone) equations in $$...$$
- Fix obvious OCR artifacts in math (e.g. broken subscripts/superscripts,
  missing braces, garbled Greek letters, broken fractions).
- If a URL appears as \\url{...}, leave it untouched.
Return ONLY the corrected text, nothing else.

TEXT:
"""


def math_fix_claude(text: str, stream_cb=None) -> str:
    """Send *text* to Claude for LaTeX math repair and return the result.

    Parameters
    ----------
    stream_cb:
        Optional callable(chunk: str) called for each streamed text chunk.
        Useful to show progress in the TUI.
    """
    try:
        import anthropic
    except ImportError:
        return text  # anthropic not installed

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return text  # no key

    client = anthropic.Anthropic(api_key=api_key)
    result_parts: list[str] = []

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": _MATH_PROMPT + text}],
    ) as stream:
        for chunk in stream.text_stream:
            result_parts.append(chunk)
            if stream_cb:
                stream_cb(chunk)

    return "".join(result_parts)


# ---------------------------------------------------------------------------
# TUI (curses)
# ---------------------------------------------------------------------------

class _Pane(NamedTuple):
    """Geometry of a single text pane."""
    top: int     # row of the title bar
    left: int    # first column
    height: int  # usable text rows (excluding title bar)
    width: int   # usable columns


def _draw_pane(
    win: curses.window,
    pane: _Pane,
    title: str,
    lines: list[str],
    scroll: int,
    active: bool,
) -> None:
    """Render a single pane (title bar + text lines)."""
    attr_title = curses.A_BOLD | curses.A_REVERSE if active else curses.A_BOLD
    # Title bar
    try:
        label = f" {title} "
        win.addstr(pane.top, pane.left, label[:pane.width].ljust(pane.width), attr_title)
    except curses.error:
        pass

    # Text lines
    for row_idx in range(pane.height):
        line_idx = scroll + row_idx
        screen_row = pane.top + 1 + row_idx
        try:
            if line_idx < len(lines):
                line = lines[line_idx]
                # Trim to pane width, pad with spaces
                displayed = line[:pane.width].ljust(pane.width)
                win.addstr(screen_row, pane.left, displayed)
            else:
                win.addstr(screen_row, pane.left, " " * pane.width)
        except curses.error:
            pass


def _status_bar(win: curses.window, rows: int, cols: int, msg: str) -> None:
    """Draw the status bar at the bottom of the screen."""
    default = "[y/↵] copy&quit  [m] math-fix  [e] edit  [q] quit  [Tab] switch pane"
    text = msg if msg else default
    try:
        win.addstr(rows - 1, 0, text[:cols].ljust(cols), curses.A_REVERSE)
    except curses.error:
        pass


def run_tui(original: str, formatted: str, auto_math: bool = False) -> tuple[str, bool]:
    """Launch the curses TUI.

    Returns
    -------
    (final_text, copied)
        *final_text* is the text to be written to the clipboard (possibly
        edited or math-fixed); *copied* is True if the user pressed y/Enter.
    """
    orig_lines = original.splitlines()
    fmt_lines = formatted.splitlines()

    result: dict = {"text": formatted, "copied": False}

    def _main(stdscr: curses.window) -> None:
        nonlocal fmt_lines

        curses.curs_set(0)
        curses.use_default_colors()
        stdscr.keypad(True)

        scroll = 0
        active_pane = 1  # 0 = left (original), 1 = right (formatted)
        status_msg = ""
        math_running = False
        _auto_math_pending = auto_math and bool(os.environ.get("ANTHROPIC_API_KEY"))

        if _auto_math_pending:
            status_msg = "Running Claude math fix…"

        while True:
            rows, cols = stdscr.getmaxyx()
            stdscr.erase()

            # Layout: two equal panes side-by-side, one status bar row at bottom
            pane_h = rows - 3   # title + text; minus header row + status bar
            half = cols // 2

            left_pane = _Pane(top=1, left=0, height=pane_h, width=half - 1)
            right_pane = _Pane(top=1, left=half, height=pane_h, width=cols - half)

            # Header
            header = " OCR Formatter — ↑↓/PgUp/PgDn scroll | Tab switch pane "
            try:
                stdscr.addstr(0, 0, header[:cols].ljust(cols), curses.A_BOLD)
            except curses.error:
                pass

            # Divider
            for r in range(1, rows - 1):
                try:
                    stdscr.addch(r, half - 1, curses.ACS_VLINE)
                except curses.error:
                    pass

            max_scroll_orig = max(0, len(orig_lines) - pane_h)
            max_scroll_fmt = max(0, len(fmt_lines) - pane_h)
            max_scroll = max(max_scroll_orig, max_scroll_fmt)
            scroll = min(scroll, max_scroll)

            _draw_pane(stdscr, left_pane, "ORIGINAL", orig_lines, scroll, active_pane == 0)
            _draw_pane(stdscr, right_pane, "FORMATTED", fmt_lines, scroll, active_pane == 1)
            _status_bar(stdscr, rows, cols, status_msg)

            stdscr.refresh()
            status_msg = ""  # clear transient messages after one frame

            # ------------------------------------------------------------------
            # Auto-math on startup
            # ------------------------------------------------------------------
            if math_running:
                math_running = False
                # Run Claude math fix (blocking in this simple implementation)
                fixed = math_fix_claude(result["text"])
                if fixed != result["text"]:
                    result["text"] = fixed
                    fmt_lines = fixed.splitlines()
                status_msg = "Math fix complete."
                continue

            if _auto_math_pending:
                _auto_math_pending = False
                math_running = True
                status_msg = "Running Claude math fix…"
                continue

            # ------------------------------------------------------------------
            # Input handling
            # ------------------------------------------------------------------
            key = stdscr.getch()

            if key in (ord('y'), ord('Y'), ord('\n'), ord('\r'), curses.KEY_ENTER):
                result["copied"] = True
                return

            if key in (ord('q'), ord('Q'), 27):  # 27 = Esc
                return

            if key == ord('\t'):
                active_pane = 1 - active_pane

            elif key in (curses.KEY_UP, ord('k')):
                scroll = max(0, scroll - 1)

            elif key in (curses.KEY_DOWN, ord('j')):
                scroll = min(max_scroll, scroll + 1)

            elif key == curses.KEY_PPAGE:
                scroll = max(0, scroll - pane_h)

            elif key == curses.KEY_NPAGE:
                scroll = min(max_scroll, scroll + pane_h)

            elif key in (ord('m'), ord('M')):
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    status_msg = "ANTHROPIC_API_KEY not set — Claude math fix unavailable."
                else:
                    math_running = True
                    status_msg = "Running Claude math fix…"

            elif key in (ord('e'), ord('E')):
                # Open formatted text in $EDITOR, reload on save
                editor = os.environ.get("EDITOR", "vi")
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                ) as tf:
                    tf.write(result["text"])
                    tmp_path = tf.name
                curses.endwin()
                subprocess.run([editor, tmp_path], check=False)
                with open(tmp_path, encoding="utf-8") as f:
                    new_text = f.read()
                os.unlink(tmp_path)
                result["text"] = new_text
                fmt_lines = new_text.splitlines()
                stdscr.refresh()
                status_msg = "Loaded edited text."

    curses.wrapper(_main)
    return result["text"], result["copied"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OCR text formatter: fixes line breaks, wraps URLs, optional Claude math fix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--stdin", action="store_true",
        help="Read input from stdin instead of the clipboard.",
    )
    p.add_argument(
        "--math", action="store_true",
        help="Automatically run Claude math fix on startup (requires ANTHROPIC_API_KEY).",
    )
    p.add_argument(
        "--no-tui", action="store_true",
        help="Non-interactive: format and write to clipboard (or stdout with --stdin), then exit.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    # --- Read input ---
    if args.stdin or not sys.stdin.isatty():
        raw = sys.stdin.read()
    else:
        try:
            raw = read_clipboard()
        except RuntimeError as exc:
            sys.exit(f"Error reading clipboard: {exc}")

    if not raw.strip():
        sys.exit("Clipboard is empty.")

    # --- Format ---
    formatted = format_ocr(raw)

    # --- Non-interactive mode ---
    if args.no_tui:
        if args.math:
            formatted = math_fix_claude(formatted)
        if args.stdin:
            print(formatted, end="")
        else:
            try:
                write_clipboard(formatted)
                print("Formatted text written to clipboard.")
            except RuntimeError as exc:
                sys.exit(f"Error writing clipboard: {exc}")
        return

    # --- TUI mode ---
    final_text, copied = run_tui(raw, formatted, auto_math=args.math)

    if copied:
        try:
            write_clipboard(final_text)
            print("Formatted text copied to clipboard.")
        except RuntimeError as exc:
            sys.exit(f"Error writing clipboard: {exc}")
    else:
        print("Aborted — clipboard unchanged.")


if __name__ == "__main__":
    main()
