"""
Asynchronous terminal input and redraw-safe output for the experiment 6
chat prototype. Derived from the `ainput` gist by k98kurz.
"""

from collections import deque
from sys import stdin, stdout
import asyncio
import os
import re
import select
import signal

try:
    import termios
    import tty
except ImportError:
    termios, tty = None, None


HISTORY_SIZE = 10
POLL_INTERVAL = 0.01
ESC_TIMEOUT = 0.01

# --- Control characters ---
CTRL_BACKSPACE = "\x08"
CTRL_DEL = "\x7f"
CTRL_D_EOF = "\x04"
CTRL_L_CLEAR = "\x0c"
ESC = "\x1b"
CTRL_E_BLOCK = "\x05"
CTRL_C_CANCEL = "\x03"

# --- Arrow sequences (bytes following ESC) ---
ARROW_UP_SEQS = ("[A", "OA", "[5A", "[1;5A", "[1;2A")
ARROW_DOWN_SEQS = ("[B", "OB", "[5B", "[1;5B", "[1;2B")
ARROW_LEFT_SEQS = ("[D", "OD", "[5D", "[1;5D", "[1;2D")
ARROW_RIGHT_SEQS = ("[C", "OC", "[5C", "[1;5C", "[1;2C")
CTRL_LEFT_SEQS = ("[5D", "[1;5D")
CTRL_RIGHT_SEQS = ("[5C", "[1;5C")
HOME_SEQS = ("[H", "OH")
END_SEQS = ("[F", "OF")
PAGE_UP_SEQS = ("[5~",)
PAGE_DOWN_SEQS = ("[6~",)

# --- ANSI terminal sequences ---
CR = "\r"
LF = "\n"
CLEAR_LINE = f"{ESC}[K"
CLEAR_ENTIRE_LINE = f"{ESC}[2K"
CLEAR_SCREEN = f"{ESC}[2J"
MOVE_DOWN_1 = f"{ESC}[1B"
MOVE_TO_LINE_START = f"{ESC}[1G"
MOVE_TO_ORIGIN = f"{ESC}[1H"
ERASE_LAST_CHAR = "\b \b"

# --- ANSI color/style codes ---
RESET = f"{ESC}[0m"
BOLD = f"{ESC}[1m"
BLACK = f"{ESC}[30m"
RED = f"{ESC}[31m"
GREEN = f"{ESC}[32m"
YELLOW = f"{ESC}[33m"
BLUE = f"{ESC}[34m"
MAGENTA = f"{ESC}[35m"
CYAN = f"{ESC}[36m"
WHITE = f"{ESC}[37m"

_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def visual_length(text: str) -> tuple[int,]:
    """Calculate the visual widths of each line in `text`, ignoring ANSI
        escape codes.
    """
    return tuple(len(_ANSI_PATTERN.sub("", line)) for line in text.split("\n"))


def ansi_move_up(n: int) -> str:
    """Return the ANSI sequence to move the cursor up `n` lines."""
    return f"{ESC}[{n}A"


class BlockRequest:
    """Sentinel returned by ``ainput`` when Ctrl-E requests block mode."""


class Console:
    """Manages an asynchronous prompt, line editing, history, and output
        that may be printed while the prompt is active.
    """
    def __init__(self, prompt: str = "you> ") -> None:
        """Initialize the console with `prompt`."""
        self.prompt = prompt
        self._line = "" # current input line
        self._lines_used = 1 # terminal lines the prompt region occupies
        self._buffer: list[str] = [""] # current input block
        self._in_block = False
        self._history: deque[str] = deque(maxlen=HISTORY_SIZE)
        self._history_index = 0
        self._active = False # whether or not we are awaiting input
        self._fallback = termios is None or tty is None or not stdin.isatty()
        self.width = 80
        self.height = 24
        self._cursor_row = 0
        self._cursor_col = 0
        self._cursor = 0 # flat cursor index into self._line
        self._cancel_requested = False
        self._fixed_lines: list[str] = []
        self._stream_line = ""
        self.fd = stdin.fileno()
        self._stream_lock = asyncio.Lock()
        if self._fallback:
            print("<Fallback mode is active>")

    async def ainput(self) -> str | None:
        """Read a line asynchronously. Returns `None` on EOF."""
        if self._fallback:
            while True:
                try:
                    return await asyncio.to_thread(input, self.prompt)
                except EOFError:
                    return None
                except KeyboardInterrupt:
                    continue

        self._line = ""
        self._cursor = 0
        self._history_index = len(self._history)
        self._active = True
        self._fixed_lines = []
        self._stream_line = ""
        saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd, termios.TCSADRAIN)

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGWINCH, self._on_resize)
        loop.add_signal_handler(signal.SIGINT, self._on_sigint)

        try:
            ts = os.get_terminal_size(self.fd)
            self.width = ts.columns
            self.height = ts.lines
            self._reset_bookkeeping()
            stdout.write(self.prompt)
            stdout.flush()

            while True:
                if self._cancel_requested:
                    self._cancel_requested = False
                    self._line = ""
                    self._cursor = 0
                    stdout.write("^C\n")
                    stdout.write(self.prompt)
                    stdout.flush()
                    self._reset_bookkeeping()
                    continue

                char = await self._read_char()

                if self._cancel_requested:
                    self._cancel_requested = False
                    self._line = ""
                    self._cursor = 0
                    stdout.write("^C\n")
                    stdout.write(self.prompt)
                    stdout.flush()
                    self._reset_bookkeeping()
                    continue

                if char is None:
                    stdout.write("\n")
                    stdout.flush()
                    return None

                if char in (CR, LF):
                    stdout.write("\n")
                    stdout.flush()

                    if self._line:
                        self._history.append(self._line)

                    return self._line

                if char in (CTRL_DEL, CTRL_BACKSPACE):
                    self._erase_char()
                    continue

                if char == CTRL_D_EOF:
                    stdout.write("\n")
                    stdout.flush()
                    return None

                if char == ESC:
                    await self._handle_escape()
                    continue

                if char == CTRL_L_CLEAR:
                    self._clear_screen()
                    self._redraw()
                    continue

                if char == CTRL_E_BLOCK:
                    self._line = ""
                    self._redraw()
                    return BlockRequest()

                if char.isprintable():
                    self._line = (
                        self._line[: self._cursor] + char
                        + self._line[self._cursor :]
                    )
                    self._cursor += 1
                    self._redraw()
        finally:
            self._active = False
            loop.remove_signal_handler(signal.SIGWINCH)
            loop.remove_signal_handler(signal.SIGINT)
            termios.tcsetattr(self.fd, termios.TCSADRAIN, saved)

    async def ainput_block(self) -> str | None:
        """Read a multi-line block asynchronously. Returns `None` on EOF
            or Ctrl-D cancel.
        """
        if self._fallback:
            lines: list[str] = []
            while True:
                try:
                    line = await asyncio.to_thread(input, self.prompt)
                except EOFError:
                    return "\n".join(lines) if lines else None
                except KeyboardInterrupt:
                    return None
                lines.append(line)

        self._buffer = [""]
        self._line = ""
        self._cursor = 0
        self._history_index = len(self._history)
        self._active = True
        self._in_block = True
        self._fixed_lines = []
        self._stream_line = ""
        saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd, termios.TCSADRAIN)

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGWINCH, self._on_resize)
        loop.add_signal_handler(signal.SIGINT, self._on_sigint)

        try:
            ts = os.get_terminal_size(self.fd)
            self.width = ts.columns
            self.height = ts.lines
            self._update_cursor()
            stdout.write(self.prompt)
            stdout.flush()

            while True:
                char = await self._read_char()

                if self._cancel_requested:
                    self._cancel_requested = False
                    self._buffer = [""]
                    self._line = ""
                    stdout.write("^C\n")
                    stdout.flush()
                    return None

                if char is None:
                    stdout.write("\n")
                    stdout.flush()
                    if self._line:
                        self._history.append(self._line)
                    return self._line

                if char in (CR, LF):
                    self._buffer.append("")
                    self._line = "\n".join(self._buffer)
                    stdout.write("\n")
                    stdout.flush()
                    self._update_cursor()
                    continue

                if char in (CTRL_DEL, CTRL_BACKSPACE):
                    self._erase_char()
                    continue

                if char == CTRL_D_EOF:
                    if not self._line:
                        stdout.write("\n")
                        stdout.flush()
                        return None
                    self._history.append(self._line)
                    stdout.write("\n")
                    stdout.flush()
                    return self._line

                if char == ESC:
                    await self._handle_escape()
                    continue

                if char == CTRL_L_CLEAR:
                    self._clear_screen()
                    self._redraw()
                    continue

                if char.isprintable():
                    self._buffer[-1] += char
                    self._line = "\n".join(self._buffer)
                    stdout.write(char)
                    stdout.flush()
                    self._update_cursor()
        finally:
            self._active = False
            self._in_block = False
            self._buffer = [""]
            loop.remove_signal_handler(signal.SIGWINCH)
            loop.remove_signal_handler(signal.SIGINT)
            termios.tcsetattr(self.fd, termios.TCSADRAIN, saved)

    async def write_line(self, text: str = "") -> None:
        """Print `text` on its own line, preserving any partially typed
            input line when the prompt is active.
        """
        if not self._active:
            stdout.write(text)
            stdout.write("\n")
            stdout.flush()
            return

        # clear line in case streaming is occurring
        async with self._stream_lock:
            self._stream_line = ""

        # erase to the beginning of the prompt line
        stdout.write(ansi_move_up(self._cursor_row))
        stdout.write(MOVE_TO_LINE_START + CLEAR_ENTIRE_LINE + CR)

        # then just fucking print the goddamn text
        stdout.write(text.strip() + "\n")

        # then fucking print the fucking prompt
        stdout.write(self.prompt)
        if self._in_block:
            stdout.write("\n".join(self._buffer))
        else:
            stdout.write(self._line)
        stdout.flush()
        self._update_cursor()

    async def write_stream(self, text: str, done: bool = False) -> None:
        """Write part of a stream of text. Expecting a token stream from
            an LLM, not chunks of multiple tokens.
        """
        if not self._active:
            stdout.write(text)
            if done:
                stdout.write("\n")
            stdout.flush()
            return

        # erase the stream line and the prompt region
        async with self._stream_lock:
            stream_lines = self._wrap_count(
                visual_length(self._stream_line)[-1]
            )
            total_lines = stream_lines + self._lines_used
            stdout.write(ansi_move_up(self._cursor_row + stream_lines))
            for _ in range(total_lines):
                stdout.write(CLEAR_ENTIRE_LINE)
                if _ < total_lines - 1:
                    stdout.write(MOVE_DOWN_1)
            if total_lines > 1:
                stdout.write(ansi_move_up(total_lines - 1))
            stdout.write(MOVE_TO_LINE_START)

            # just a single fucking newline
            if not text.strip() and "\n" in text:
                stdout.write(self._stream_line + "\n")
                self._stream_line = ""

            # if text contains a new line
            elif "\n" in text:
                parts = text.split("\n")
                stdout.write(self._stream_line + parts[0] + "\n")
                if len(parts) > 2:
                    for p in parts[1:-1]:
                        stdout.write(p + "\n")
                self._stream_line = parts[-1] if len(parts) > 0 else ""

            # normal fucking text
            else:
                self._stream_line += text
                stdout.write(self._stream_line)

        # print an extra newline when fucking done
        if done:
            stdout.write("\n")

        # then fucking print the fucking prompt
        stdout.write("\n" + self.prompt)
        if self._in_block:
            stdout.write("\n".join(self._buffer))
        else:
            stdout.write(self._line)
        stdout.flush()
        self._update_cursor()

    def _wrap_count(self, width: int) -> int:
        """Number of terminal lines required for `width` visual chars."""
        return (width + self.width - 1) // self.width if width > 0 else 0

    def _position_after_width(self, width: int) -> tuple[int, int]:
        """Give the `(row, col)` of the cursor position after writing
            `width` visual characters from the top-left corner.
        """
        if width == 0:
            return 0, 0
        if width % self.width == 0:
            return width // self.width, 0
        return (width - 1) // self.width, width % self.width

    def _reset_bookkeeping(self) -> None:
        """Reset region size and cursor to a prompt-only, empty-line state."""
        prompt_col = visual_length(self.prompt)[-1]
        self._lines_used = self._wrap_count(prompt_col)
        self._cursor_row, self._cursor_col = self._position_after_width(
            prompt_col
        )

    def close(self) -> None:
        """Clear input state. Terminal attributes are restored by
            `ainput` when it exits.
        """
        self._active = False
        self._fixed_lines = []
        self._stream_line = ""

    async def _read_char(self) -> str | None:
        """Read one Unicode character without blocking.
            Returns `None` on EOF.
        """
        while True:
            if self._cancel_requested:
                return CTRL_C_CANCEL
            ready, _, _ = select.select([stdin], [], [], 0)
            if not ready:
                await asyncio.sleep(POLL_INTERVAL)
                continue
            first = os.read(self.fd, 1)
            if not first:
                return None
            code = first[0]
            if code < 0x80:
                n = 1
            elif 0xC0 <= code < 0xE0:
                n = 2
            elif 0xE0 <= code < 0xF0:
                n = 3
            else:
                n = 4
            data = first
            while len(data) < n:
                ready, _, _ = select.select([stdin], [], [], 0)
                if not ready:
                    return "\ufffd"
                b = os.read(self.fd, 1)
                if not b:
                    return "\ufffd"
                data += b
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return "\ufffd"

    async def _read_escape_sequence(self) -> str:
        """Read the escape sequence bytes following an escape character."""
        data = b""

        ready, _, _ = select.select([stdin], [], [], ESC_TIMEOUT)
        if not ready:
            return ""
        byte = os.read(self.fd, 1)
        if not byte:
            return ""
        data += byte

        if byte == b"[":
            while True:
                ready, _, _ = select.select([stdin], [], [], ESC_TIMEOUT)
                if not ready:
                    break
                b = os.read(self.fd, 1)
                if not b:
                    break
                data += b
                if b >= b"@":
                    break
        elif byte == b"O":
            ready, _, _ = select.select([stdin], [], [], ESC_TIMEOUT)
            if ready:
                b = os.read(self.fd, 1)
                if b:
                    data += b

        return data.decode(errors="replace")

    async def _handle_escape(self) -> None:
        """Handle arrow keys, ignoring all other escape sequences."""
        sequence = await self._read_escape_sequence()

        if sequence in ARROW_UP_SEQS:
            self._recall(-1)
        elif sequence in ARROW_DOWN_SEQS:
            self._recall(1)
        elif sequence in ARROW_LEFT_SEQS:
            if sequence in CTRL_LEFT_SEQS:
                self._move_cursor_word_left()
            else:
                self._move_cursor_left()
        elif sequence in ARROW_RIGHT_SEQS:
            if sequence in CTRL_RIGHT_SEQS:
                self._move_cursor_word_right()
            else:
                self._move_cursor_right()
        elif sequence in HOME_SEQS:
            self._move_cursor_home()
        elif sequence in END_SEQS:
            self._move_cursor_end()
        elif sequence in PAGE_UP_SEQS:
            self._recall(-1)
        elif sequence in PAGE_DOWN_SEQS:
            self._recall(1)

    def _recall(self, direction: int) -> None:
        """Replace the current line with the previous or next history
            entry. `direction` is -1 for older and 1 for newer.
        """
        if not self._history:
            return

        index = self._history_index + direction

        if index < 0:
            return

        if index >= len(self._history):
            self._history_index = len(self._history)
            self._line = ""
            self._cursor = 0
            if self._in_block:
                self._buffer = [""]
        else:
            self._history_index = index
            self._line = self._history[index]
            if self._in_block:
                self._buffer = self._line.split("\n") if self._line else [""]
            else:
                self._cursor = len(self._line)

        self._redraw()

    def _move_cursor_left(self) -> None:
        """Move cursor left one position, if possible."""
        if not self._in_block and self._cursor > 0:
            self._cursor -= 1
            self._redraw()

    def _move_cursor_right(self) -> None:
        """Move cursor right one position, if possible."""
        if not self._in_block and self._cursor < len(self._line):
            self._cursor += 1
            self._redraw()

    def _move_cursor_home(self) -> None:
        """Move cursor to the beginning of the line, if possible."""
        if not self._in_block and self._cursor != 0:
            self._cursor = 0
            self._redraw()

    def _move_cursor_end(self) -> None:
        """Move cursor to the end of the line, if possible."""
        if not self._in_block and self._cursor != len(self._line):
            self._cursor = len(self._line)
            self._redraw()

    def _move_cursor_word_left(self) -> None:
        """Move cursor left to the previous word boundary."""
        if self._in_block:
            return
        if self._cursor == 0:
            return
        line = self._line
        pos = self._cursor
        if pos > 0 and not line[pos - 1].isspace():
            while pos > 0 and not line[pos - 1].isspace():
                pos -= 1
        else:
            while pos > 0 and line[pos - 1].isspace():
                pos -= 1
            while pos > 0 and not line[pos - 1].isspace():
                pos -= 1
        self._cursor = pos
        self._redraw()

    def _move_cursor_word_right(self) -> None:
        """Move cursor right to the next word boundary."""
        if self._in_block:
            return
        if self._cursor >= len(self._line):
            return
        line = self._line
        pos = self._cursor
        if pos < len(line) and line[pos].isspace():
            while pos < len(line) and line[pos].isspace():
                pos += 1
            while pos < len(line) and not line[pos].isspace():
                pos += 1
        else:
            while pos < len(line) and not line[pos].isspace():
                pos += 1
            while pos < len(line) and line[pos].isspace():
                pos += 1
        self._cursor = pos
        self._redraw()

    def _clear_screen(self) -> None:
        """Clear the terminal screen, if possible."""
        if self._fallback:
            return
        stdout.write(CLEAR_SCREEN)
        stdout.write(MOVE_TO_ORIGIN)
        stdout.flush()
        self._reset_bookkeeping()

    def _on_sigint(self, *args) -> None:
        """Handle Ctrl-C by requesting cancellation."""
        self._cancel_requested = True

    def _on_resize(self) -> None:
        """Flag that the terminal was resized so the input loop redraws."""
        ts = os.get_terminal_size(self.fd)
        self.width = ts.columns
        self.height = ts.lines
        self._redraw()

    def _erase_char(self) -> None:
        """Remove the character before the cursor and redraw."""
        if self._in_block:
            if self._buffer[-1]:
                self._buffer[-1] = self._buffer[-1][:-1]
            elif len(self._buffer) > 1:
                self._buffer.pop()
            self._line = "\n".join(self._buffer)
            self._cursor = len(self._line)
            self._redraw()
        else:
            if self._cursor == 0:
                return
            self._line = self._line[:self._cursor - 1] + self._line[self._cursor:]
            self._cursor -= 1
            self._redraw()

    def _redraw(self) -> None:
        """Redraw the prompt and current line."""
        total, cursor_row, cursor_col = self._layout()
        to_clear = max(self._lines_used, total)

        if self._cursor_row:
            stdout.write(ansi_move_up(self._cursor_row))
        stdout.write(MOVE_TO_LINE_START + CLEAR_ENTIRE_LINE)

        for i in range(to_clear):
            stdout.write(CLEAR_LINE)
            if i < to_clear - 1:
                stdout.write(MOVE_DOWN_1)

        if to_clear > 1:
            stdout.write(ansi_move_up(to_clear - 1))
        stdout.write(MOVE_TO_LINE_START)
        stdout.write(self.prompt + self._line)

        end_row, end_col = self._position_after_width(
            visual_length(self.prompt + self._line)[-1]
        )
        move_up = end_row - cursor_row
        if move_up > 0:
            stdout.write(ansi_move_up(move_up))
        if not self._in_block:
            if end_col > cursor_col:
                stdout.write(f"{ESC}[{end_col - cursor_col}D")
            elif end_col < cursor_col:
                stdout.write(f"{ESC}[{cursor_col - end_col}C")

        self._lines_used = total
        self._cursor_row = cursor_row
        self._cursor_col = cursor_col
        stdout.flush()

    def _layout(self) -> tuple[int, int, int]:
        """Screen geometry of `prompt + line`.
            Returns `(total_lines, cursor_row, cursor_col)`.
        """
        widths = visual_length(self.prompt + self._line)
        wraps = tuple(self._wrap_count(w) for w in widths)
        total = sum(wraps)
        before = self._line[: self._cursor]
        line_index = before.count("\n")
        chunk = before[before.rfind("\n") + 1 :]
        col = visual_length(chunk)[-1]
        if line_index == 0:
            col += visual_length(self.prompt)[-1]
        base_row = sum(wraps[:line_index])
        if col == 0:
            cursor_row, cursor_col = base_row, 0
        elif col % self.width == 0:
            cursor_row, cursor_col = base_row + col // self.width, 0
        else:
            cursor_row, cursor_col = (
                base_row + (col - 1) // self.width, col % self.width
            )
        return total, cursor_row, cursor_col

    def _update_cursor(self) -> None:
        """Resync bookkeeping after the region was fully rewritten."""
        self._lines_used, self._cursor_row, self._cursor_col = self._layout()
