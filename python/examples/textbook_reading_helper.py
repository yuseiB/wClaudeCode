#!/usr/bin/env python3
"""Textbook Reading Helper — AI-powered assistant for scientific textbooks.

Helps with four core reading tasks:
  1. /define  <term>    — Precise definition + intuition + related terms
  2. /symbol  <char>    — Map a symbol to its physical/mathematical meaning
  3. /equation          — Break an equation down term-by-term
  4. /concept <name>    — Explore a concept and its connections in the field

Free-form questions are also accepted — just type and press Enter.

Image support: use /image <path> to load a page scan or diagram before your
next question, so Claude can read equations and figures directly.

Usage:
    python textbook_reading_helper.py
    python textbook_reading_helper.py --image chapter3_page47.png
    ANTHROPIC_API_KEY=sk-... python textbook_reading_helper.py
"""

import argparse
import base64
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional readline for better line-editing experience
# ---------------------------------------------------------------------------
try:
    import readline  # noqa: F401  (side-effect: enables arrow keys / history)
except ImportError:
    pass  # Windows — fine, just no readline

import anthropic

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert scientific textbook reading assistant specialising in \
physics, mathematics, and information science. \
You help readers deeply understand textbook content through four focused \
modes of analysis. Be rigorous, precise, and build on concepts already \
discussed in the session.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE 1 — TERM DEFINITIONS  (triggered by /define)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When defining a term, always provide:
• Formal definition (mathematical or scientific)
• Plain-language intuition
• Domain context — what sub-field uses it, and why it matters
• Notation variants found across textbooks
• Related terms: prerequisite concepts, closely related terms, \
  terms often confused with this one
• A minimal worked example or usage in an equation/theorem

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE 2 — SYMBOL / CHARACTER MAPPING  (triggered by /symbol)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When mapping a symbol, always provide a structured table:

| Symbol | Quantity / Object | Typical units | Constraints / range |
|--------|-------------------|---------------|---------------------|

Then explain:
• Standard conventions — which disciplines use this symbol and for what
• Conflicts — other quantities that share the same symbol in other fields
• How this symbol appears in key equations in this domain
• Subscripts/superscripts/modifier conventions (e.g. $\\hat{x}$, $\\vec{v}$, $x'$)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE 3 — EQUATION TERM ANALYSIS  (triggered by /equation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When analysing an equation, always provide:
1. Name and purpose — what physical/mathematical law does this encode?
2. Term-by-term breakdown as a numbered list:
   Term → symbol(s) → meaning → SI units → role in the equation
3. What the equation is "saying" in one sentence
4. Assumptions and validity domain (when does it break down?)
5. Special/limiting cases (e.g. linear approximation, high-T limit)
6. Connections to other fundamental equations (derive from / leads to)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE 4 — KEY CONCEPTS & CONNECTIONS  (triggered by /concept)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When exploring a concept, always provide:
• Core idea — one precise statement
• Physical / mathematical intuition
• Concept map:
  ◦ Prerequisite concepts (what you must understand first)
  ◦ Closely related concepts (same level of abstraction)
  ◦ Derived concepts (what this concept enables)
• Chapter/section connections — where does this appear again in typical texts?
• Cross-field connections — how does this concept appear in neighbouring \
  disciplines (e.g. the same formalism used in thermodynamics AND \
  information theory)?
• Common misconceptions and how to resolve them
• Canonical references or theorems named after this concept

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATTING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Use LaTeX for all mathematics: $inline$ and $$display block$$
• Use Markdown tables for symbol mappings and structured data
• Use numbered lists for equation term breakdowns
• Use bold for the first occurrence of a new term being defined
• If an image is provided, read all equations and figures from it directly
• If context is ambiguous, ask a clarifying question before answering

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SESSION MEMORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Remember all terms, symbols, and equations discussed in this session
• Build on established definitions — do not repeat what is already known
• Explicitly flag when a new concept connects back to an earlier one
• Track the conceptual thread so the reader sees how ideas build on \
  each other across sections and chapters
"""

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
MAGENTA = "\033[1;35m"
RED = "\033[1;31m"


def c(text: str, code: str) -> str:
    """Wrap text in an ANSI colour code (no-op when not a TTY)."""
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{RESET}"


# ---------------------------------------------------------------------------
# Banner and help text
# ---------------------------------------------------------------------------

BANNER = f"""
{c('╔══════════════════════════════════════════════════════════╗', CYAN)}
{c('║', CYAN)}  {c('Scientific Textbook Reading Helper', BOLD)}                     {c('║', CYAN)}
{c('║', CYAN)}  Powered by Claude Opus 4.6 with adaptive thinking        {c('║', CYAN)}
{c('╚══════════════════════════════════════════════════════════╝', CYAN)}
"""

HELP_TEXT = f"""
{c('Commands', BOLD)}
  {c('/define  <term>', GREEN)}     Precise definition + intuition + related terms
  {c('/symbol  <char>', GREEN)}     Map a symbol to its physical/mathematical meaning
  {c('/equation [eq]', GREEN)}      Break an equation down term by term
  {c('/concept [name]', GREEN)}     Explore a concept and its connections in the field
  {c('/image   <path>', YELLOW)}     Load a textbook page image for the next question
  {c('/clear', YELLOW)}              Clear conversation history and start fresh
  {c('/help', DIM)}                Show this help
  {c('/quit  or  /exit', DIM)}      Exit

{c('Tips', BOLD)}
  • Type any free-form question — you don't need a command prefix.
  • Load an image first with /image, then ask about it in your next message.
  • Claude remembers everything in the session — use that context.
  • For LaTeX equations, paste them as-is: e.g.  /equation E = mc^2
"""


def print_banner() -> None:
    print(BANNER)
    print(c("Type /help for commands, or just ask a question.\n", DIM))


def print_help() -> None:
    print(HELP_TEXT)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def load_image(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for a local image file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    ext = p.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image type '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
        )
    media_type = MEDIA_TYPES[ext]
    data = base64.standard_b64encode(p.read_bytes()).decode("utf-8")
    return data, media_type


# ---------------------------------------------------------------------------
# Core helper class
# ---------------------------------------------------------------------------


class TextbookHelper:
    """Stateful textbook-reading assistant backed by the Claude API."""

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                c(
                    "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
                    "Export it before running: export ANTHROPIC_API_KEY=sk-...",
                    RED,
                )
            )
            sys.exit(1)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.messages: list[dict] = []
        self._pending_image: tuple[str, str] | None = None  # (data, media_type)

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------

    def queue_image(self, path: str) -> str:
        """Load an image so it is attached to the next user message."""
        try:
            data, media_type = load_image(path)
            self._pending_image = (data, media_type)
            name = Path(path).name
            return c(f"Image queued: {name}  ({media_type})", GREEN)
        except (FileNotFoundError, ValueError) as exc:
            return c(f"Error: {exc}", RED)

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_content(self, text: str) -> list | str:
        """Wrap text (and any queued image) into a messages-API content block."""
        if self._pending_image:
            data, media_type = self._pending_image
            self._pending_image = None  # consume it
            return [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                },
                {"type": "text", "text": text},
            ]
        return text

    # ------------------------------------------------------------------
    # Streaming response
    # ------------------------------------------------------------------

    def ask(self, text: str) -> None:
        """Send *text* to Claude and stream the response to stdout."""
        content = self._build_content(text)
        self.messages.append({"role": "user", "content": content})

        # System prompt with prompt caching (TTL = 1 h; re-sent but only
        # charged at cached-input rate after the first request)
        system = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]

        print()  # blank line before response
        full_response = ""
        in_thinking = False

        try:
            with self.client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=8192,
                thinking={"type": "adaptive"},
                system=system,
                messages=self.messages,
            ) as stream:
                for event in stream:
                    etype = event.type

                    if etype == "content_block_start":
                        btype = event.content_block.type
                        if btype == "thinking":
                            in_thinking = True
                            print(c("⟨thinking⟩", DIM), flush=True)
                        elif btype == "text" and in_thinking:
                            in_thinking = False
                            print(c("⟨/thinking⟩\n", DIM), flush=True)

                    elif etype == "content_block_delta":
                        delta = event.delta
                        if delta.type == "thinking_delta":
                            # Show thinking in dim so users can follow reasoning
                            print(c(delta.thinking, DIM), end="", flush=True)
                        elif delta.type == "text_delta":
                            print(delta.text, end="", flush=True)
                            full_response += delta.text

                    elif etype == "content_block_stop" and in_thinking:
                        in_thinking = False
                        print(c("\n⟨/thinking⟩\n", DIM), flush=True)

        except anthropic.AuthenticationError:
            print(c("\nAuthentication error: check your ANTHROPIC_API_KEY.", RED))
            self.messages.pop()  # undo the user message we just added
            return
        except anthropic.RateLimitError:
            print(c("\nRate limit reached. Please wait a moment and try again.", RED))
            self.messages.pop()
            return
        except anthropic.APIConnectionError:
            print(c("\nNetwork error. Check your internet connection.", RED))
            self.messages.pop()
            return
        except anthropic.APIStatusError as exc:
            print(c(f"\nAPI error {exc.status_code}: {exc.message}", RED))
            self.messages.pop()
            return

        print()  # newline after streamed response
        self.messages.append({"role": "assistant", "content": full_response})

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self.messages.clear()
        self._pending_image = None
        print(c("Session cleared — starting fresh.", YELLOW))

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------

    def run(self, initial_image: str | None = None) -> None:
        """Start the interactive read-eval-print loop."""
        print_banner()

        if initial_image:
            print(self.queue_image(initial_image))
            print(
                c(
                    "Image loaded. Ask your first question to analyse it.\n",
                    DIM,
                )
            )

        while True:
            try:
                raw = input(c("▶ ", CYAN)).strip()
            except (EOFError, KeyboardInterrupt):
                print(c("\nGoodbye!", DIM))
                break

            if not raw:
                continue

            # ── Commands ────────────────────────────────────────────────
            if raw.startswith("/"):
                parts = raw.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit"):
                    print(c("Goodbye!", DIM))
                    break

                elif cmd == "/help":
                    print_help()

                elif cmd == "/clear":
                    self.clear()

                elif cmd == "/image":
                    if arg:
                        print(self.queue_image(arg))
                    else:
                        print(c("Usage: /image <path-to-image>", RED))

                elif cmd == "/define":
                    if not arg:
                        print(c("Usage: /define <term>", RED))
                        continue
                    prompt = (
                        f"Please define the term **{arg}** using MODE 1 — "
                        "TERM DEFINITIONS guidelines."
                    )
                    self.ask(prompt)

                elif cmd == "/symbol":
                    if not arg:
                        print(c("Usage: /symbol <symbol or symbol name>", RED))
                        continue
                    prompt = (
                        f"Map the symbol **{arg}** to its physical/mathematical "
                        "meaning using MODE 2 — SYMBOL / CHARACTER MAPPING guidelines."
                    )
                    self.ask(prompt)

                elif cmd == "/equation":
                    if arg:
                        prompt = (
                            f"Analyse this equation term by term using MODE 3 — "
                            f"EQUATION TERM ANALYSIS guidelines:\n\n{arg}"
                        )
                    else:
                        prompt = (
                            "Please describe or paste the equation you want analysed. "
                            "I will break it down term by term following MODE 3 — "
                            "EQUATION TERM ANALYSIS guidelines."
                        )
                    self.ask(prompt)

                elif cmd == "/concept":
                    if arg:
                        prompt = (
                            f"Explore the key concept **{arg}** using MODE 4 — "
                            "KEY CONCEPTS & CONNECTIONS guidelines."
                        )
                    else:
                        prompt = (
                            "What concept would you like to explore? "
                            "Describe it and I will map its connections to "
                            "other concepts and the broader field using MODE 4 — "
                            "KEY CONCEPTS & CONNECTIONS guidelines."
                        )
                    self.ask(prompt)

                else:
                    print(
                        c(
                            f"Unknown command '{cmd}'. Type /help for available commands.",
                            RED,
                        )
                    )

            # ── Free-form question ───────────────────────────────────────
            else:
                self.ask(raw)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-powered scientific textbook reading assistant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image",
        metavar="PATH",
        help="Load a textbook page image to analyse at startup.",
    )
    args = parser.parse_args()

    helper = TextbookHelper()
    helper.run(initial_image=args.image)


if __name__ == "__main__":
    main()
