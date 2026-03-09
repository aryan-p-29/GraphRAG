"""
parsers/__init__.py
───────────────────
BaseParser ABC and ParserDispatcher (the registry + factory).

Usage:
    from parsers import ParserDispatcher

    parser = ParserDispatcher.get_parser(".py")
    nodes, edges = parser.parse(source_bytes, file_path="src/foo.py")
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path

from config import LANGUAGE_REGISTRY
from graph_schema import NodeData, EdgeData


class BaseParser(ABC):
    """Abstract base class every language parser must implement."""

    @abstractmethod
    def parse(
        self,
        source: bytes,
        file_path: str,
    ) -> tuple[list[NodeData], list[EdgeData]]:
        """
        Parse *source* (raw file bytes) and return graph elements.

        Parameters
        ----------
        source:     Raw bytes of the source file.
        file_path:  Relative path string used as part of node IDs.

        Returns
        -------
        (nodes, edges)  Lists of NodeData and EdgeData.
        """


class ParserDispatcher:
    """
    Registry-based factory that maps file extensions to parser instances.
    Grammars are loaded lazily — only extensions actually encountered are imported.
    """

    _cache: dict[str, BaseParser] = {}

    @classmethod
    def get_parser(cls, extension: str) -> BaseParser:
        """
        Return a (cached) parser instance for *extension* (e.g. '.py').
        Raises KeyError if the extension is not in LANGUAGE_REGISTRY.
        """
        ext = extension.lower()
        if ext not in cls._cache:
            if ext not in LANGUAGE_REGISTRY:
                raise KeyError(
                    f"No parser registered for extension {ext!r}. "
                    f"Supported: {sorted(LANGUAGE_REGISTRY)}"
                )
            cfg = LANGUAGE_REGISTRY[ext]
            module = importlib.import_module(cfg["module"])
            parser_cls = getattr(module, cfg["class"])
            cls._cache[ext] = parser_cls()
        return cls._cache[ext]

    @classmethod
    def supports(cls, path: str | Path) -> bool:
        """Return True if this path's extension has a registered parser."""
        return Path(path).suffix.lower() in LANGUAGE_REGISTRY
