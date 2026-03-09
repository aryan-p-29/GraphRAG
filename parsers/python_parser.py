"""
parsers/python_parser.py
────────────────────────
Tree-sitter-based parser for Python source files.

Extracts:
  Nodes : File, Class, Function, Method, Module
  Edges : DEFINES, HAS_METHOD, CALLS, IMPORTS, INHERITS

Post-parse:
  FuzzyMethodResolver — resolves dynamic `obj.method()` call patterns by
  scanning all known Method/Function nodes and rerouting unresolved CALLS
  edges when there is an unambiguous (or best-effort) target.
"""

from __future__ import annotations

import re

from typing import Optional

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

from graph_schema import NodeData, EdgeData
from parsers import BaseParser

PY_LANGUAGE = Language(tspython.language())


def _node_text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_docstring(body_node: Node, src: bytes) -> Optional[str]:
    """Return the first expression_statement string literal child as docstring."""
    for child in body_node.children:
        if child.type == "expression_statement":
            for grandchild in child.children:
                if grandchild.type == "string":
                    raw = _node_text(grandchild, src)
                    return re.sub(r'^[\'\"]{1,3}|[\'\"]{1,3}$', '', raw).strip()
    return None


def _file_node(file_path: str) -> NodeData:
    import os
    return NodeData(
        id=file_path,
        label="File",
        properties={"path": file_path, "name": os.path.basename(file_path), "language": "python"},
        embed_text=f"Python file: {file_path}",
    )


# ─────────────────────────────────────────────────────────────────────────────


class PythonParser(BaseParser):
    """Parses Python files using Tree-sitter and emits NodeData / EdgeData."""

    def __init__(self) -> None:
        self._parser = Parser(PY_LANGUAGE)

    # ── Decorator unwrapping helper ───────────────────────────────────────────
    @staticmethod
    def _unwrap_definition(node: Node) -> Node | None:
        """
        Given a node that is either a `function_definition`, `class_definition`,
        or `decorated_definition`, return the innermost `function_definition`
        or `class_definition`.

        A `decorated_definition` in tree-sitter looks like:
            decorator*
            function_definition | class_definition

        This helper peels any number of `decorated_definition` wrappers and
        returns the bare definition node, or None if not found.
        """
        if node.type in ("function_definition", "class_definition"):
            return node
        if node.type == "decorated_definition":
            for child in node.children:
                result = PythonParser._unwrap_definition(child)
                if result is not None:
                    return result
        return None

    # ──────────────────────────────────────────────────────────────────────────
    def parse(self, source: bytes, file_path: str) -> tuple[list[NodeData], list[EdgeData]]:
        tree = self._parser.parse(source)
        root = tree.root_node

        nodes: list[NodeData] = []
        edges: list[EdgeData] = []

        file_node = _file_node(file_path)
        nodes.append(file_node)

        self._walk_module(root, source, file_path, nodes, edges)



        return nodes, edges

    # ──────────────────────────────────────────────────────────────────────────
    def _walk_module(
        self,
        root: Node,
        src: bytes,
        file_path: str,
        nodes: list[NodeData],
        edges: list[EdgeData],
    ) -> None:
        for child in root.children:
            if child.type == "import_statement":
                self._handle_import(child, src, file_path, nodes, edges)
            elif child.type == "import_from_statement":
                self._handle_import_from(child, src, file_path, nodes, edges)
            else:
                # Handles plain function_definition, class_definition, and
                # decorated_definition (any nesting depth) uniformly.
                defn = self._unwrap_definition(child)
                if defn is not None:
                    if defn.type == "class_definition":
                        self._handle_class(defn, src, file_path, nodes, edges)
                    elif defn.type == "function_definition":
                        self._handle_function(defn, src, file_path, nodes, edges)

    # ── Import handling ───────────────────────────────────────────────────────
    def _add_module(self, name: str, file_path: str, nodes: list, edges: list) -> None:
        mod_id = f"module::{name}"
        if not any(n.id == mod_id for n in nodes):
            nodes.append(NodeData(id=mod_id, label="Module", properties={"name": name}))
        edges.append(EdgeData(type="IMPORTS", source_id=file_path, target_id=mod_id))

    def _handle_import(self, node: Node, src: bytes, file_path: str, nodes: list, edges: list) -> None:
        for child in node.children:
            if child.type in ("dotted_name", "aliased_import"):
                name_node = child if child.type == "dotted_name" else child.children[0]
                name = _node_text(name_node, src).split(".")[0]
                self._add_module(name, file_path, nodes, edges)

    def _handle_import_from(self, node: Node, src: bytes, file_path: str, nodes: list, edges: list) -> None:
        for child in node.children:
            if child.type == "dotted_name":
                name = _node_text(child, src).split(".")[0]
                self._add_module(name, file_path, nodes, edges)
                break

    # ── Class handling ────────────────────────────────────────────────────────
    def _handle_class(self, node: Node, src: bytes, file_path: str, nodes: list, edges: list) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        class_name = _node_text(name_node, src)
        class_id = f"{file_path}::{class_name}"

        body = node.child_by_field_name("body")
        docstring = _get_docstring(body, src) if body else None

        nodes.append(
            NodeData(
                id=class_id,
                label="Class",
                properties={
                    "name": class_name,
                    "file_path": file_path,
                    "lineno": node.start_point[0] + 1,
                    "docstring": docstring or "",
                    "language": "python",
                },
                embed_text=f"Class {class_name}: {docstring or ''}",
            )
        )
        edges.append(EdgeData(type="DEFINES", source_id=file_path, target_id=class_id))

        # Inheritance
        superclasses = node.child_by_field_name("superclasses")
        if superclasses:
            for arg in superclasses.children:
                if arg.type in ("identifier", "dotted_name"):
                    parent_name = _node_text(arg, src)
                    parent_id = f"{file_path}::{parent_name}"
                    edges.append(EdgeData(type="INHERITS", source_id=class_id, target_id=parent_id))

        # Methods — use _unwrap_definition so decorated methods are never missed
        if body:
            for item in body.children:
                defn = self._unwrap_definition(item)
                if defn is not None and defn.type == "function_definition":
                    self._handle_method(defn, src, file_path, class_id, class_name, nodes, edges)

    # ── Function handling ─────────────────────────────────────────────────────
    def _handle_function(self, node: Node, src: bytes, file_path: str, nodes: list, edges: list) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        fn_name = _node_text(name_node, src)
        fn_id = f"{file_path}::{fn_name}"

        params_node = node.child_by_field_name("parameters")
        sig = f"def {fn_name}{_node_text(params_node, src) if params_node else '()'}"

        body = node.child_by_field_name("body")
        docstring = _get_docstring(body, src) if body else None

        nodes.append(
            NodeData(
                id=fn_id,
                label="Function",
                properties={
                    "name": fn_name,
                    "file_path": file_path,
                    "lineno": node.start_point[0] + 1,
                    "docstring": docstring or "",
                    "signature": sig,
                    "language": "python",
                },
                embed_text=f"Function {sig}: {docstring or ''}",
            )
        )
        edges.append(EdgeData(type="DEFINES", source_id=file_path, target_id=fn_id))

        if body:
            self._extract_calls(body, src, fn_id, file_path, edges)

    # ── Method handling ───────────────────────────────────────────────────────
    def _handle_method(
        self,
        node: Node,
        src: bytes,
        file_path: str,
        class_id: str,
        class_name: str,
        nodes: list,
        edges: list,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        method_name = _node_text(name_node, src)
        method_id = f"{class_id}::{method_name}"

        params_node = node.child_by_field_name("parameters")
        sig = f"def {method_name}{_node_text(params_node, src) if params_node else '()'}"

        body = node.child_by_field_name("body")
        docstring = _get_docstring(body, src) if body else None

        nodes.append(
            NodeData(
                id=method_id,
                label="Method",
                properties={
                    "name": method_name,
                    "parent_name": class_name,
                    "file_path": file_path,
                    "lineno": node.start_point[0] + 1,
                    "docstring": docstring or "",
                    "signature": sig,
                    "language": "python",
                },
                embed_text=f"Method {class_name}.{sig}: {docstring or ''}",
            )
        )
        edges.append(EdgeData(type="HAS_METHOD", source_id=class_id, target_id=method_id))

        if body:
            self._extract_calls(body, src, method_id, file_path, edges)

    # ── Call extraction ───────────────────────────────────────────────────────
    def _extract_calls(self, body: Node, src: bytes, caller_id: str, file_path: str, edges: list) -> None:
        _SCOPE_TYPES = frozenset({
            "function_definition",
            "class_definition",
            "decorated_definition",
        })

        # We rely on pure LIFO stack traversal. No seen set. No memory leaks.
        stack: list[Node] = list(body.children)

        while stack:
            node = stack.pop()

            # --- ADD THIS PROBE ---
            if "inventory_prediction" in caller_id:
                print(f"DEBUG WALK: Visiting -> {node.type}")
                if node.type == "call":
                    fn_node = node.child_by_field_name("function")
                    print(f"   -> CALL FOUND! Function child type: {fn_node.type if fn_node else 'None'}")
            # ----------------------

            # ── Rule 1: skip nested scopes ────────────────────────────────
            if node.type in _SCOPE_TYPES:
                continue

            # ── Rule 2: emit CALLS edge for every call expression ─────────
            if node.type == "call":
                fn_node = node.child_by_field_name("function")
                if fn_node is not None:
                    if fn_node.type == "identifier":
                        callee = _node_text(fn_node, src)
                    elif fn_node.type == "attribute":
                        attr_node = fn_node.child_by_field_name("attribute")
                        callee = (
                            _node_text(attr_node, src)
                            if attr_node is not None
                            else _node_text(fn_node, src).split(".")[-1]
                        )
                    else:
                        raw    = _node_text(fn_node, src)
                        callee = raw.replace("self.", "").split(".")[0]

                    callee_id = f"{file_path}::{callee}"
                    edges.append(EdgeData(
                        type="CALLS",
                        source_id=caller_id,
                        target_id=callee_id,
                    ))

            # ── Rule 3: continue into children ────────────────────────────
            for child in node.children:
                stack.append(child)
