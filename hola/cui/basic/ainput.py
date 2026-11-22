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
ESC = "\x1b"
CTRL_E_BLOCK = "\x05"
CTRL_C_CANCEL = "\x03"

# --- Arrow sequences (bytes following ESC) ---
ARROW_UP_SEQS = ("[A", "OA")
ARROW_DOWN_SEQS = ("[B", "OB")

# --- ANSI terminal sequences ---
CR = "\r"
LF = "\n"
CLEAR_LINE = f"{ESC}[K"
CLEAR_ENTIRE_LINE = f"{ESC}[2K"
MOVE_DOWN_1 = f"{ESC}[1B"
MOVE_TO_LINE_START = f"{ESC}[1G"
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


def visual_length(text: str) -> int:
    """Return the visual width of `text`, ignoring ANSI escape codes."""
    return len(_ANSI_PATTERN.sub("", text))


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
        self._lines_used = 1 # how many wrapped lines displayed for self._line
        self._buffer: list[str] = [""] # current input block
        self._in_block = False
        self._history: deque[str] = deque(maxlen=HISTORY_SIZE)
        self._history_index = 0
        self._active = False # whether or not we are awaiting input
        self._fallback = termios is None or tty is None or not stdin.isatty()
        self.width = 80
        self.height = 24
        self._col = 0
        self._cancel_requested = False
        self._fixed_lines: list[str] = []
        self._stream_line = ""
        self.fd = stdin.fileno()
        self._stream_lock = asyncio.Lock()

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
            prompt_total = visual_length(self.prompt)
            self._lines_used = prompt_total // self.width + 1
            self._col = prompt_total % self.width
            stdout.write(self.prompt)
            stdout.flush()

            while True:
                if self._cancel_requested:
                    self._cancel_requested = False
                    self._line = ""
                    stdout.write("^C\n")
                    stdout.write(self.prompt)
                    stdout.flush()
                    self._col = visual_length(self.prompt) % self.width
                    continue

                char = await self._read_char()

                if self._cancel_requested:
                    self._cancel_requested = False
                    self._line = ""
                    stdout.write("^C\n")
                    stdout.write(self.prompt)
                    stdout.flush()
                    self._col = visual_length(self.prompt) % self.width
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

                if char == CTRL_E_BLOCK:
                    self._line = ""
                    self._redraw()
                    return BlockRequest()

                if char.isprintable():
                    if self._col + 1 >= self.width:
                        self._lines_used += 1
                        self._col = 0
                    else:
                        self._col += 1
                    self._line += char
                    stdout.write(char)
                    stdout.flush()
        finally:
            self._active = False
            loop.remove_signal_handler(signal.SIGWINCH)
            loop.remove_signal_handler(signal.SIGINT)
            termios.tcsetattr(self.fd, termios.TCSADRAIN, saved)

    async def ainput_block(self) -> str | None:
        """Read a multi-line block asynchronously. Returns `None` on EOF
            or Ctrl-C cancel.
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
                if self._cancel_requested:
                    self._cancel_requested = False
                    self._buffer = [""]
                    self._line = ""
                    stdout.write("\n")
                    stdout.flush()
                    return None

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

                if char.isprintable():
                    self._buffer[-1] += char
                    self._line = "\n".join(self._buffer)
                    stdout.write(char)
                    stdout.flush()
                    self._update_cursor()
        finally:
            self._active = False
            self._in_block = False
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
        if self._lines_used > 1:
            stdout.write(ansi_move_up(self._lines_used))
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

        # erase to the beginning of the most recent streamed line
        async with self._stream_lock:
            if self._lines_used > 1:
                stdout.write(ansi_move_up(self._lines_used))
            lines_to_clear = self._line_visual(self._stream_line)
            lines_to_clear += self._line_visual(self.prompt + self._line)
            if self._in_block:
                lines_to_clear += self._lines_used - 1
            if lines_to_clear > 1:
                for _ in range(lines_to_clear - 1):
                    stdout.write(CLEAR_ENTIRE_LINE)
                    stdout.write(ansi_move_up(1))
            stdout.write(MOVE_TO_LINE_START + CLEAR_ENTIRE_LINE)

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

    def _line_visual(self, line: str) -> int:
        """Count visual lines a single logical line occupies at the
            current terminal width, accounting for wrapping.
        """
        L = visual_length(line)
        if L == 0:
            return 1
        if L % self.width == 0:
            return L // self.width
        return (L - 1) // self.width + 1

    def close(self) -> None:
        """Clear input state. Terminal attributes are restored by
            `ainput` when it exits.
        """
        self._active = False
        self._fixed_lines = []
        self._stream_line = ""

    async def _read_char(self) -> str | None:
        """Read one character without blocking. Returns `None` on EOF."""
        while True:
            if self._cancel_requested:
                return CTRL_C_CANCEL
            ready, _, _ = select.select([stdin], [], [], 0)
            if ready:
                data = os.read(self.fd, 1)
                if not data:
                    return None
                return data.decode(errors="replace")
            await asyncio.sleep(POLL_INTERVAL)

    async def _read_escape_sequence(self) -> str:
        """Read up to two bytes following an escape character."""
        data = b""

        for _ in range(2):
            ready, _, _ = select.select([stdin], [], [], ESC_TIMEOUT)
            if not ready:
                break
            byte = os.read(self.fd, 1)
            if not byte:
                break
            data += byte

        return data.decode(errors="replace")

    async def _handle_escape(self) -> None:
        """Handle arrow keys, ignoring all other escape sequences."""
        sequence = await self._read_escape_sequence()

        if sequence in ARROW_UP_SEQS:
            self._recall(-1)
        elif sequence in ARROW_DOWN_SEQS:
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
            if self._in_block:
                self._buffer = [""]
        else:
            self._history_index = index
            self._line = self._history[index]
            if self._in_block:
                self._buffer = self._line.split("\n") if self._line else [""]

        self._redraw()

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
        """Remove the last character from the line and the display."""
        if not self._line:
            return

        if self._in_block:
            if self._buffer[-1]:
                self._buffer[-1] = self._buffer[-1][:-1]
            elif len(self._buffer) > 1:
                self._buffer.pop()
            self._line = "\n".join(self._buffer)
            self._redraw()
        else:
            old_total = visual_length(self.prompt) + visual_length(self._line)
            self._line = self._line[:-1]
            new_total = visual_length(self.prompt) + visual_length(self._line)

            old_line = old_total // self.width
            new_line = new_total // self.width

            if new_line < old_line:
                self._redraw()
            else:
                self._col = new_total % self.width
                stdout.write(ERASE_LAST_CHAR)
        stdout.flush()

    def _redraw(self) -> None:
        """Redraw the prompt and current line."""
        old_line_num = self._lines_used - 1
        old_lines_used = self._lines_used

        if self._in_block:
            line_num, col = self._cursor_position()
            new_lines_used = line_num + 1
        else:
            total = visual_length(self.prompt) + visual_length(self._line)
            line_num = total // self.width
            col = total % self.width
            new_lines_used = line_num + 1

        to_clear = max(old_lines_used, new_lines_used)

        if old_line_num > 0:
            stdout.write(ansi_move_up(old_line_num))
        stdout.write(CR)

        for i in range(to_clear):
            stdout.write(CLEAR_LINE)
            if i < to_clear - 1:
                stdout.write(MOVE_DOWN_1)

        if to_clear > 1:
            stdout.write(ansi_move_up(to_clear - 1))
        stdout.write(CR)

        self._col = col
        self._lines_used = new_lines_used

        stdout.write(self.prompt + self._line)
        stdout.flush()

    def _cursor_position(self) -> tuple[int, int]:
        """Compute (line_num, col) for the cursor at the end of
            the multi-line buffer.
        """
        lines = [self.prompt + self._buffer[0]] + self._buffer[1:]
        line_num = 0
        for line in lines[:-1]:
            L = visual_length(line)
            if L == 0:
                line_num += 1
            elif L % self.width == 0:
                line_num += L // self.width
            else:
                line_num += (L - 1) // self.width + 1
        last = lines[-1]
        L = visual_length(last)
        if L == 0:
            return line_num, 0
        if L % self.width == 0:
            return line_num + L // self.width, 0
        return line_num + (L - 1) // self.width, L % self.width

    def _update_cursor(self) -> None:
        """Update cursor position from the current buffer state."""
        line_num, col = self._cursor_position()
        self._col = col
        self._lines_used = line_num + 1
