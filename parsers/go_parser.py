"""
parsers/go_parser.py
─────────────────────
Tree-sitter-based parser for Go source files.

Extracts:
  Nodes : File, Function, Method, Struct, Interface, Module
  Edges : DEFINES, HAS_METHOD, CALLS, IMPORTS, IMPLEMENTS
"""

from __future__ import annotations

import os
from typing import Optional

import tree_sitter_go as tsgo
from tree_sitter import Language, Parser, Node

from graph_schema import NodeData, EdgeData
from parsers import BaseParser

GO_LANGUAGE = Language(tsgo.language())


def _node_text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_comment(node: Node, src: bytes) -> Optional[str]:
    """Look for a preceding comment_block (Go doc comments appear as siblings)."""
    # Not directly accessible from the node; return None for now.
    return None


class GoParser(BaseParser):
    """Parses Go source files using Tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(GO_LANGUAGE)

    # ──────────────────────────────────────────────────────────────────────────
    def parse(self, source: bytes, file_path: str) -> tuple[list[NodeData], list[EdgeData]]:
        tree = self._parser.parse(source)
        root = tree.root_node

        nodes: list[NodeData] = []
        edges: list[EdgeData] = []

        nodes.append(NodeData(
            id=file_path, label="File",
            properties={"path": file_path, "name": os.path.basename(file_path), "language": "go"},
            embed_text=f"Go file: {file_path}",
        ))

        self._walk(root, source, file_path, nodes, edges)
        return nodes, edges

    # ──────────────────────────────────────────────────────────────────────────
    def _walk(self, root: Node, src: bytes, file_path: str, nodes: list, edges: list) -> None:
        for child in root.children:
            t = child.type
            if t == "import_declaration":
                self._handle_import(child, src, file_path, nodes, edges)
            elif t == "function_declaration":
                self._handle_function(child, src, file_path, nodes, edges)
            elif t == "method_declaration":
                self._handle_method(child, src, file_path, nodes, edges)
            elif t == "type_declaration":
                self._handle_type_decl(child, src, file_path, nodes, edges)

    # ── Import ────────────────────────────────────────────────────────────────
    def _handle_import(self, node: Node, src: bytes, file_path: str, nodes: list, edges: list) -> None:
        for child in node.children:
            if child.type == "import_spec_list":
                for spec in child.children:
                    if spec.type == "import_spec":
                        for part in spec.children:
                            if part.type == "interpreted_string_literal":
                                raw = _node_text(part, src).strip('"')
                                name = raw.split("/")[-1]
                                mod_id = f"module::{name}"
                                if not any(n.id == mod_id for n in nodes):
                                    nodes.append(NodeData(id=mod_id, label="Module", properties={"name": name, "path": raw}))
                                edges.append(EdgeData(type="IMPORTS", source_id=file_path, target_id=mod_id))

    # ── Function ──────────────────────────────────────────────────────────────
    def _handle_function(self, node: Node, src: bytes, file_path: str, nodes: list, edges: list) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        fn_name = _node_text(name_node, src)
        fn_id = f"{file_path}::{fn_name}"

        params = node.child_by_field_name("parameters")
        result = node.child_by_field_name("result")
        sig = f"func {fn_name}"
        if params:
            sig += _node_text(params, src)
        if result:
            sig += " " + _node_text(result, src)

        nodes.append(NodeData(
            id=fn_id, label="Function",
            properties={"name": fn_name, "file_path": file_path,
                        "lineno": node.start_point[0] + 1, "docstring": "", "signature": sig, "language": "go"},
            embed_text=f"Go function {sig}",
        ))
        edges.append(EdgeData(type="DEFINES", source_id=file_path, target_id=fn_id))

        body = node.child_by_field_name("body")
        if body:
            self._extract_calls(body, src, fn_id, file_path, edges)

    # ── Method (receiver function) ────────────────────────────────────────────
    def _handle_method(self, node: Node, src: bytes, file_path: str, nodes: list, edges: list) -> None:
        name_node = node.child_by_field_name("name")
        recv_node = node.child_by_field_name("receiver")
        if not name_node:
            return

        method_name = _node_text(name_node, src)
        # Extract receiver type (the struct name)
        receiver_type = "Unknown"
        if recv_node:
            for child in recv_node.children:
                if child.type == "parameter_declaration":
                    for part in child.children:
                        if part.type in ("type_identifier", "pointer_type"):
                            raw = _node_text(part, src).lstrip("*")
                            receiver_type = raw
                            break

        parent_id = f"{file_path}::{receiver_type}"
        method_id = f"{parent_id}::{method_name}"

        params = node.child_by_field_name("parameters")
        result = node.child_by_field_name("result")
        sig = f"func ({receiver_type}) {method_name}"
        if params:
            sig += _node_text(params, src)
        if result:
            sig += " " + _node_text(result, src)

        nodes.append(NodeData(
            id=method_id, label="Method",
            properties={"name": method_name, "parent_name": receiver_type, "file_path": file_path,
                        "lineno": node.start_point[0] + 1, "docstring": "", "signature": sig, "language": "go"},
            embed_text=f"Go method {sig}",
        ))
        edges.append(EdgeData(type="HAS_METHOD", source_id=parent_id, target_id=method_id))

        body = node.child_by_field_name("body")
        if body:
            self._extract_calls(body, src, method_id, file_path, edges)

    # ── Type declarations (struct / interface) ────────────────────────────────
    def _handle_type_decl(self, node: Node, src: bytes, file_path: str, nodes: list, edges: list) -> None:
        for spec in node.children:
            if spec.type == "type_spec":
                name_node = spec.child_by_field_name("name")
                type_node = spec.child_by_field_name("type")
                if not name_node or not type_node:
                    continue
                type_name = _node_text(name_node, src)
                type_kind = type_node.type  # struct_type, interface_type, ...
                type_id = f"{file_path}::{type_name}"

                if type_kind == "struct_type":
                    nodes.append(NodeData(
                        id=type_id, label="Struct",
                        properties={"name": type_name, "file_path": file_path,
                                    "lineno": spec.start_point[0] + 1, "docstring": "", "language": "go"},
                        embed_text=f"Go struct {type_name}",
                    ))
                elif type_kind == "interface_type":
                    nodes.append(NodeData(
                        id=type_id, label="Interface",
                        properties={"name": type_name, "file_path": file_path,
                                    "lineno": spec.start_point[0] + 1, "docstring": "", "language": "go"},
                        embed_text=f"Go interface {type_name}",
                    ))
                else:
                    # Generic type alias
                    nodes.append(NodeData(
                        id=type_id, label="TypeAlias",
                        properties={"name": type_name, "file_path": file_path,
                                    "lineno": spec.start_point[0] + 1,
                                    "target_type": _node_text(type_node, src), "language": "go"},
                        embed_text=f"Go type alias {type_name}",
                    ))

                edges.append(EdgeData(type="DEFINES", source_id=file_path, target_id=type_id))

    # ── Call extraction ───────────────────────────────────────────────────────
    def _extract_calls(self, body: Node, src: bytes, caller_id: str, file_path: str, edges: list) -> None:
        visited: set[int] = set()

        def recurse(n: Node) -> None:
            if id(n) in visited:
                return
            visited.add(id(n))
            if n.type == "call_expression":
                fn_node = n.child_by_field_name("function")
                if fn_node:
                    raw = _node_text(fn_node, src).split(".")[0]
                    callee_id = f"{file_path}::{raw}"
                    edges.append(EdgeData(type="CALLS", source_id=caller_id, target_id=callee_id))
            for child in n.children:
                recurse(child)

        recurse(body)
