"""
Ingestion utilities for the Sep Text Manifold project.

The functions in this module traverse a directory tree, read text
files and yield their contents.  They normalise all inputs to UTF‑8
strings and preserve file identifiers so that later stages can
associate computed metrics back to the original documents.

You may wish to extend this module to handle additional file types
(such as JSON or CSV) or to stream very large files in chunks
instead of reading them into memory.
"""

from __future__ import annotations

import os
import pathlib
from typing import Iterable, Tuple, Optional, Generator


def ingest_directory(directory: str, *, extensions: Optional[Iterable[str]] = None) -> Generator[Tuple[str, str, str], None, None]:
    """Recursively traverse *directory* and yield `(file_id, path, text)` tuples.

    Parameters
    ----------
    directory:
        Root directory to scan for files.
    extensions:
        Optional iterable of allowed file extensions (without the dot).  If
        provided, only files with one of these extensions (case
        insensitive) will be processed.  If ``None`` (the default) then
        all files are considered.

    Yields
    ------
    file_id : str
        A unique identifier derived from the file's relative path.  This
        can be used as a key in downstream tables.
    path : str
        The absolute path to the file on disk.
    text : str
        The file contents decoded as UTF‑8.  Any decoding errors are
        replaced with the Unicode replacement character ("\ufffd").
    """
    root = pathlib.Path(directory)
    if not root.is_dir():
        raise ValueError(f"{directory} is not a directory")
    exts = None
    if extensions is not None:
        exts = {ext.lower() for ext in extensions}
    file_paths = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())
    for path in file_paths:
        if exts is not None:
            if path.suffix:
                suffix = path.suffix.lstrip(".").lower()
                if suffix not in exts:
                    continue
            else:
                continue
        try:
            data = path.read_text(encoding="utf-8", errors="replace")
        except (UnicodeDecodeError, OSError):
            # Skip files we cannot decode
            continue
        # Use relative path from root as file_id
        file_id = str(path.relative_to(root)).replace(os.sep, "/")
        yield file_id, str(path), data
