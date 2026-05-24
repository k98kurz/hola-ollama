"""
Bundled tools for agent filesystem and memory interaction.
"""

from pathlib import Path
from hola.tools.core import tool
from hola.types import SessionProtocol
from hola.ollama.exceptions import ToolExecutionError
import shutil
import stat
import time


@tool(name="memorize", description="Store a value in session memory")
def memorize(key: str, value: object, session: SessionProtocol) -> None:
    session.memorize(key, value)


@tool(name="recall", description="Retrieve a value from session memory")
def recall(key: str, session: SessionProtocol) -> object:
    return session.recall(key)


@tool(name="read", description="Read file contents as text")
def read(path: str, session: SessionProtocol) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = Path(session.cwd) / path_obj

    if not session.get_file_permission(path_obj).read:
        raise ToolExecutionError(f"Read permission denied: {path_obj}")

    try:
        return path_obj.read_text(errors='strict')
    except FileNotFoundError:
        raise ToolExecutionError(f"File not found: {path_obj}")
    except PermissionError:
        raise ToolExecutionError(f"Permission denied: {path_obj}")


@tool(name="write", description="Write content to file")
def write(path: str, content: str, session: SessionProtocol) -> None:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = Path(session.cwd) / path_obj

    if not session.get_file_permission(path_obj.parent).read:
        raise ToolExecutionError(
            f"Read permission denied for parent: {path_obj.parent}"
        )

    if not session.get_file_permission(path_obj.parent).write:
        raise ToolExecutionError(
            f"Write permission denied for parent: {path_obj.parent}"
        )

    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if not session.get_file_permission(path_obj).write:
        raise ToolExecutionError(f"Write permission denied: {path_obj}")

    path_obj.write_text(content)


@tool(name="rm", description="Remove file or directory")
def rm(path: str, session: SessionProtocol, r: bool = False) -> None:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = Path(session.cwd) / path_obj

    if not path_obj.exists():
        raise ToolExecutionError(f"Path not found: {path_obj}")

    if not session.get_file_permission(path_obj).write:
        raise ToolExecutionError(f"Write permission denied: {path_obj}")

    if path_obj.is_file():
        path_obj.unlink()
    elif path_obj.is_dir():
        if r:
            shutil.rmtree(path_obj)
        else:
            try:
                path_obj.rmdir()
            except OSError:
                raise ToolExecutionError(
                    f"Directory not empty (use -r): {path_obj}"
                )


@tool(name="mv", description="Move/rename file or directory")
def mv(src: str, dst: str, session: SessionProtocol) -> None:
    src_obj = Path(src)
    dst_obj = Path(dst)

    if not src_obj.is_absolute():
        src_obj = Path(session.cwd) / src_obj
    if not dst_obj.is_absolute():
        dst_obj = Path(session.cwd) / dst_obj

    if not session.get_file_permission(src_obj).write:
        raise ToolExecutionError(f"Write permission denied: {src_obj}")

    if not session.get_file_permission(dst_obj.parent).read:
        raise ToolExecutionError(
            f"Read permission denied for parent: {dst_obj.parent}"
        )

    if not session.get_file_permission(dst_obj.parent).write:
        raise ToolExecutionError(
            f"Write permission denied for parent: {dst_obj.parent}"
        )

    if not src_obj.exists():
        raise ToolExecutionError(f"Source not found: {src_obj}")

    shutil.move(src_obj, dst_obj)


@tool(name="cp", description="Copy file or directory")
def cp(src: str, dst: str, session: SessionProtocol, r: bool = False) -> None:
    src_obj = Path(src)
    dst_obj = Path(dst)

    if not src_obj.is_absolute():
        src_obj = Path(session.cwd) / src_obj
    if not dst_obj.is_absolute():
        dst_obj = Path(session.cwd) / dst_obj

    if not session.get_file_permission(src_obj).read:
        raise ToolExecutionError(f"Read permission denied: {src_obj}")

    if not session.get_file_permission(dst_obj.parent).read:
        raise ToolExecutionError(
            f"Read permission denied for parent: {dst_obj.parent}"
        )

    if not session.get_file_permission(dst_obj.parent).write:
        raise ToolExecutionError(
            f"Write permission denied for parent: {dst_obj.parent}"
        )

    if not src_obj.exists():
        raise ToolExecutionError(f"Source not found: {src_obj}")

    if src_obj.is_file():
        shutil.copy2(src_obj, dst_obj)
    elif src_obj.is_dir():
        if r:
            shutil.copytree(src_obj, dst_obj)
        else:
            raise ToolExecutionError(
                f"Directory copy requires -r flag: {src_obj}"
            )


@tool(name="grep", description="Search for pattern in files or data")
def grep(
        pattern: str,
        path: str = ".",
        data: str | list | dict | None = None,
        session: SessionProtocol = None,
        r: bool = False,
        n: bool = False,
        rn: bool = False,
        A: int = 0,
        B: int = 0,
    ) -> list[str]:
    results = []

    if data is not None:
        search_data = data
        if isinstance(search_data, list):
            search_data = "\n".join(str(item) for item in search_data)
        elif isinstance(search_data, dict):
            search_data = str(search_data)
        elif not isinstance(search_data, str):
            search_data = str(search_data)

        lines = search_data.split("\n")
        for line_num, line in enumerate(lines):
            if pattern in line:
                if n or rn:
                    results.append(f"{line_num + 1}:{line}")
                else:
                    results.append(line)

                if A > 0:
                    for i in range(1, A + 1):
                        if line_num + i < len(lines):
                            if n or rn:
                                results.append(
                                    f"{line_num + i + 1}-{lines[line_num + i]}"
                                )
                            else:
                                results.append(lines[line_num + i])

                if B > 0:
                    for i in range(1, B + 1):
                        if line_num - i >= 0:
                            if n or rn:
                                results.append(
                                    f"{line_num - i + 1}-{lines[line_num - i]}"
                                )
                            else:
                                results.append(lines[line_num - i])
    else:
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = Path(session.cwd) / path_obj

        if not session.get_file_permission(path_obj).read:
            raise ToolExecutionError(f"Read permission denied: {path_obj}")

        if not path_obj.exists():
            raise ToolExecutionError(f"Path not found: {path_obj}")

        if path_obj.is_file():
            files_to_search = [path_obj]
        elif path_obj.is_dir():
            if not r:
                files_to_search = [
                    p for p in path_obj.iterdir()
                    if p.is_file() and session.get_file_permission(p).read
                ]
            else:
                files_to_search = []

                def collect_files(dir_path: Path) -> None:
                    for entry in dir_path.iterdir():
                        if not session.get_file_permission(entry).read:
                            continue
                        if entry.is_file():
                            files_to_search.append(entry)
                        elif entry.is_dir():
                            collect_files(entry)

                collect_files(path_obj)
        else:
            return results

        for file_path in files_to_search:
            try:
                content = file_path.read_text()
            except (PermissionError, UnicodeDecodeError):
                continue

            lines = content.split("\n")

            if len(files_to_search) == 1:
                use_prefix = False
                relative_path = file_path.name
            else:
                use_prefix = True
                relative_path = str(file_path.relative_to(path_obj))

            for line_num, line in enumerate(lines):
                if pattern in line:
                    match_index = len(results)

                    if use_prefix:
                        if n or rn:
                            results.append(f"{relative_path}:{line_num + 1}:{line}")
                        else:
                            results.append(f"{relative_path}:{line}")
                    else:
                        if n or rn:
                            results.append(f"{line_num + 1}:{line}")
                        else:
                            results.append(line)

                    if A > 0:
                        for i in range(1, A + 1):
                            if line_num + i < len(lines):
                                if use_prefix:
                                    if n or rn:
                                        results.append(
                                            f"{relative_path}:{line_num + i + 1}-"
                                            f"{lines[line_num + i]}"
                                        )
                                    else:
                                        results.append(
                                            f"{relative_path}:{lines[line_num + i]}"
                                        )
                                else:
                                    if n or rn:
                                        results.append(
                                            f"{line_num + i + 1}-"
                                            f"{lines[line_num + i]}"
                                        )
                                    else:
                                        results.append(lines[line_num + i])

                    if B > 0:
                        context_lines = []
                        for i in range(1, B + 1):
                            if line_num - i >= 0:
                                if use_prefix:
                                    if n or rn:
                                        context_lines.append(
                                            f"{relative_path}:{line_num - i + 1}-"
                                            f"{lines[line_num - i]}"
                                        )
                                    else:
                                        context_lines.append(
                                            f"{relative_path}:{lines[line_num - i]}"
                                        )
                                else:
                                    if n or rn:
                                        context_lines.append(
                                            f"{line_num - i + 1}-"
                                            f"{lines[line_num - i]}"
                                        )
                                    else:
                                        context_lines.append(lines[line_num - i])
                        results[match_index:match_index] = context_lines

    return results


@tool(name="ls", description="List files in directory")
def ls(
        path: str = ".",
        session: SessionProtocol = None,
        l: bool = False,
        a: bool = False,
        R: bool = False,
        t: bool = False,
        S: bool = False,
        r: bool = False,
    ) -> list[str]:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = Path(session.cwd) / path_obj

    if not session.get_file_permission(path_obj).read:
        raise ToolExecutionError(f"Read permission denied: {path_obj}")

    if not path_obj.exists():
        raise ToolExecutionError(f"Path not found: {path_obj}")

    if not path_obj.is_dir():
        raise ToolExecutionError(f"Not a directory: {path_obj}")

    results = []

    def list_directory(dir_path: Path, prefix: str = "") -> None:
        entries = sorted(dir_path.iterdir(), key=lambda x: x.name.lower())

        if not a:
            entries = [e for e in entries if not e.name.startswith(".")]

        if t:
            entries.sort(key=lambda x: x.stat().st_mtime, reverse=not r)
        elif S:
            entries.sort(key=lambda x: x.stat().st_size, reverse=not r)
        elif r:
            entries.reverse()

        for entry in entries:
            if l:
                try:
                    stat_info = entry.stat()
                    perms = stat.filemode(stat_info.st_mode)
                    size = stat_info.st_size
                    mtime = time.strftime(
                        "%b %d %H:%M",
                        time.localtime(stat_info.st_mtime)
                    )
                    results.append(f"{perms}  {size:6d}  {mtime}  {entry.name}")
                except OSError:
                    results.append(f"??????????  ??????  ?????????  {entry.name}")
            else:
                results.append(entry.name)

        if R:
            for entry in sorted(
                    entries,
                    key=lambda x: x.name.lower()
                ):
                if entry.is_dir() and (a or not entry.name.startswith(".")):
                    results.append(f"{prefix}{entry.name}:")
                    new_prefix = f"{prefix}{entry.name}/"
                    list_directory(entry, new_prefix)

    results.append(f"{path_obj}:")
    list_directory(path_obj)

    return results


@tool(name="edit", description="Replace text in file")
def edit(
        path: str,
        old_string: str,
        new_string: str,
        session: SessionProtocol,
        count: int = 1,
        skip: int = 0,
    ) -> None:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = Path(session.cwd) / path_obj

    if not session.get_file_permission(path_obj).read:
        raise ToolExecutionError(f"Read permission denied: {path_obj}")

    if not session.get_file_permission(path_obj).write:
        raise ToolExecutionError(f"Write permission denied: {path_obj}")

    try:
        content = path_obj.read_text(errors='strict')
    except FileNotFoundError:
        raise ToolExecutionError(f"File not found: {path_obj}")
    except PermissionError:
        raise ToolExecutionError(f"Permission denied: {path_obj}")

    parts = content.split(old_string)
    if len(parts) < 2:
        raise ToolExecutionError(f"Old string not found: {old_string[:50]}...")

    occurrences = len(parts) - 1
    start_index = skip
    end_index = occurrences if count == 0 else min(start_index + count, occurrences)

    new_parts = []
    for i in range(len(parts)):
        new_parts.append(parts[i])
        if i < len(parts) - 1:
            if i >= start_index and i < end_index:
                new_parts.append(new_string)
            else:
                new_parts.append(old_string)

    new_content = "".join(new_parts)
    path_obj.write_text(new_content)


@tool(name="append", description="Append content to file")
def append(path: str, content: str, session: SessionProtocol) -> None:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = Path(session.cwd) / path_obj

    if path_obj.exists():
        if not session.get_file_permission(path_obj).write:
            raise ToolExecutionError(f"Write permission denied: {path_obj}")

    if not session.get_file_permission(path_obj.parent).read:
        raise ToolExecutionError(
            f"Read permission denied for parent: {path_obj.parent}"
        )

    if not session.get_file_permission(path_obj.parent).write:
        raise ToolExecutionError(
            f"Write permission denied for parent: {path_obj.parent}"
        )

    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "a") as f:
        f.write(content)


@tool(name="find", description="Find files matching criteria")
def find(
        path: str = ".",
        session: SessionProtocol = None,
        name: str | None = None,
        type: str | None = None,
        mindepth: int = 0,
        maxdepth: int | None = None,
    ) -> list[str]:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = Path(session.cwd) / path_obj

    if not session.get_file_permission(path_obj).read:
        raise ToolExecutionError(f"Read permission denied: {path_obj}")

    if not path_obj.exists():
        raise ToolExecutionError(f"Path not found: {path_obj}")

    if not path_obj.is_dir():
        raise ToolExecutionError(f"Not a directory: {path_obj}")

    results = []

    def scan_directory(dir_path: Path, current_depth: int, prefix: str = "") -> None:
        for entry in sorted(dir_path.iterdir(), key=lambda x: x.name.lower()):
            if not session.get_file_permission(entry).read:
                continue

            entry_type = None
            if entry.is_file():
                entry_type = "f"
            elif entry.is_dir():
                entry_type = "d"
            elif entry.is_symlink():
                entry_type = "l"

            relative_path = prefix + entry.name

            if current_depth >= mindepth:
                if name is None or entry.match(name):
                    if type is None or entry_type == type:
                        results.append(relative_path)

            if entry.is_dir():
                new_prefix = relative_path + "/"
                if maxdepth is None or current_depth + 1 <= maxdepth:
                    scan_directory(entry, current_depth + 1, new_prefix)

    scan_directory(path_obj, 1)

    return results
