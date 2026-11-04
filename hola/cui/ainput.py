"""
Asynchronous terminal input and redraw-safe output for the experiment 6
chat prototype. Derived from the `ainput` gist by k98kurz.
"""

from collections import deque
from sys import stdin, stdout
import asyncio
import os
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


class BlockRequest:
    """Sentinel returned by ``ainput`` when Ctrl-E requests block mode."""


class Console:
    """Manages an asynchronous prompt, line editing, history, and
        output that may be printed while the prompt is active.
    """
    def __init__(self, prompt: str = "you> ") -> None:
        """Initialize the console with `prompt`."""
        self.prompt = prompt
        self._line = ""
        self._buffer: list[str] = [""]
        self._in_block = False
        self._history: deque[str] = deque(maxlen=HISTORY_SIZE)
        self._history_index = 0
        self._active = False
        self._fallback = termios is None or tty is None or not stdin.isatty()
        self.width = 80
        self._col = 0
        self._line_num = 0
        self._resized = False
        self._lines_used = 1
        self._cancel_requested = False

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
        fd = stdin.fileno()
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd, termios.TCSADRAIN)

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGWINCH, self._on_resize)
        loop.add_signal_handler(signal.SIGINT, self._on_sigint)

        try:
            self.width = os.get_terminal_size(fd).columns
            prompt_total = len(self.prompt)
            self._line_num = prompt_total // self.width
            self._col = prompt_total % self.width
            self._resized = False
            self._lines_used = self._line_num + 1
            stdout.write(self.prompt)
            stdout.flush()

            while True:
                if self._resized:
                    self._resized = False
                    self.width = os.get_terminal_size(fd).columns
                    self._redraw()
                    continue

                if self._cancel_requested:
                    self._cancel_requested = False
                    self._line = ""
                    stdout.write("^C\n")
                    stdout.write(self.prompt)
                    stdout.flush()
                    self._line_num += 1
                    self._col = len(self.prompt) % self.width
                    self._lines_used = self._line_num + 1
                    continue

                char = await self._read_char()

                if self._cancel_requested:
                    self._cancel_requested = False
                    self._line = ""
                    stdout.write("^C\n")
                    stdout.write(self.prompt)
                    stdout.flush()
                    self._line_num += 1
                    self._col = len(self.prompt) % self.width
                    self._lines_used = self._line_num + 1
                    continue

                if char is None:
                    stdout.write("\n")
                    stdout.flush()
                    return None

                if char in ("\r", "\n"):
                    stdout.write("\n")
                    stdout.flush()

                    if self._line:
                        self._history.append(self._line)

                    return self._line

                if char in ("\x7f", "\x08"):
                    self._erase_char()
                    continue

                if char == "\x04":
                    stdout.write("\n")
                    stdout.flush()
                    return None

                if char == "\x1b":
                    await self._handle_escape()
                    continue

                if char == "\x05":
                    self._line = ""
                    self._redraw()
                    return BlockRequest()

                if char.isprintable():
                    if self._col + 1 >= self.width:
                        self._line_num += 1
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
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

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
        fd = stdin.fileno()
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd, termios.TCSADRAIN)

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGWINCH, self._on_resize)
        loop.add_signal_handler(signal.SIGINT, self._on_sigint)

        try:
            self.width = os.get_terminal_size(fd).columns
            self._update_cursor()
            stdout.write(self.prompt)
            stdout.flush()

            while True:
                if self._resized:
                    self._resized = False
                    self.width = os.get_terminal_size(fd).columns
                    self._redraw()
                    continue

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

                if char in ("\r", "\n"):
                    self._buffer.append("")
                    self._line = "\n".join(self._buffer)
                    stdout.write("\n")
                    stdout.flush()
                    self._update_cursor()
                    continue

                if char in ("\x7f", "\x08"):
                    self._erase_char()
                    continue

                if char == "\x04":
                    if not self._line:
                        stdout.write("\n")
                        stdout.flush()
                        return None
                    self._history.append(self._line)
                    stdout.write("\n")
                    stdout.flush()
                    return self._line

                if char == "\x1b":
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
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    def write_line(self, text: str = "") -> None:
        """Print `text` on its own line, preserving any partially typed
            input line when the prompt is active.
        """
        if self._active:
            old_lines_used = self._lines_used
            old_line_num = self._line_num

            if old_line_num > 0:
                stdout.write(f"\033[{old_line_num}A")
            stdout.write("\r")

            for i in range(old_lines_used):
                stdout.write("\033[K")
                if i < old_lines_used - 1:
                    stdout.write("\033[1B")

            if old_lines_used > 1:
                stdout.write(f"\033[{old_lines_used - 1}A")
            stdout.write("\r")

            stdout.write(text + "\n")

            if self._in_block:
                self._update_cursor()
            else:
                total = len(self.prompt) + len(self._line)
                self._line_num = total // self.width
                self._col = total % self.width
                self._lines_used = self._line_num + 1
            stdout.write(self.prompt + self._line)
        else:
            stdout.write(text + "\n")

        stdout.flush()

    def close(self) -> None:
        """Clear input state. Terminal attributes are restored by
            `ainput` when it exits.
        """
        self._active = False

    async def _read_char(self) -> str | None:
        """Read one character without blocking. Returns `None` on EOF."""
        fd = stdin.fileno()

        while True:
            if self._cancel_requested:
                return "\x03"
            ready, _, _ = select.select([stdin], [], [], 0)
            if ready:
                data = os.read(fd, 1)
                if not data:
                    return None
                return data.decode(errors="replace")
            await asyncio.sleep(POLL_INTERVAL)

    async def _read_escape_sequence(self) -> str:
        """Read up to two bytes following an escape character."""
        fd = stdin.fileno()
        data = b""

        for _ in range(2):
            ready, _, _ = select.select([stdin], [], [], ESC_TIMEOUT)
            if not ready:
                break
            byte = os.read(fd, 1)
            if not byte:
                break
            data += byte

        return data.decode(errors="replace")

    async def _handle_escape(self) -> None:
        """Handle arrow keys, ignoring all other escape sequences."""
        sequence = await self._read_escape_sequence()

        if sequence in ("[A", "OA"):
            self._recall(-1)
        elif sequence in ("[B", "OB"):
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
        self._resized = True

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
            old_total = len(self.prompt) + len(self._line)
            self._line = self._line[:-1]
            new_total = len(self.prompt) + len(self._line)

            old_line = old_total // self.width
            new_line = new_total // self.width

            if new_line < old_line:
                self._redraw()
            else:
                self._line_num = new_line
                self._col = new_total % self.width
                stdout.write("\b \b")
        stdout.flush()

    def _redraw(self) -> None:
        """Redraw the prompt and current line."""
        old_line_num = self._line_num
        old_lines_used = self._lines_used

        if self._in_block:
            line_num, col = self._cursor_position()
            new_lines_used = line_num + 1
        else:
            total = len(self.prompt) + len(self._line)
            line_num = total // self.width
            col = total % self.width
            new_lines_used = line_num + 1

        to_clear = max(old_lines_used, new_lines_used)

        if old_line_num > 0:
            stdout.write(f"\033[{old_line_num}A")
        stdout.write("\r")

        for i in range(to_clear):
            stdout.write("\033[K")
            if i < to_clear - 1:
                stdout.write("\033[1B")

        if to_clear > 1:
            stdout.write(f"\033[{to_clear - 1}A")
        stdout.write("\r")

        self._line_num = line_num
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
            L = len(line)
            if L == 0:
                line_num += 1
            elif L % self.width == 0:
                line_num += L // self.width
            else:
                line_num += (L - 1) // self.width + 1
        last = lines[-1]
        L = len(last)
        if L == 0:
            return line_num, 0
        if L % self.width == 0:
            return line_num + L // self.width, 0
        return line_num + (L - 1) // self.width, L % self.width

    def _update_cursor(self) -> None:
        """Update cursor position from the current buffer state."""
        line_num, col = self._cursor_position()
        self._line_num = line_num
        self._col = col
        self._lines_used = line_num + 1
