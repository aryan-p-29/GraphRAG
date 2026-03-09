"""
parsers/javascript_parser.py
─────────────────────────────
Tree-sitter-based parser for JavaScript and TypeScript source files.

Extracts:
  Nodes : File, Class, Function, Method, Interface, TypeAlias, Module
  Edges : DEFINES, HAS_METHOD, CALLS, IMPORTS, INHERITS
"""

from __future__ import annotations

import os
from typing import Optional

from tree_sitter import Language, Parser, Node
from graph_schema import NodeData, EdgeData
from parsers import BaseParser


def _load_language(ext: str) -> Language:
    if ext == ".ts":
        import tree_sitter_typescript as tsts
        return Language(tsts.language_typescript())
    else:
        import tree_sitter_javascript as tsjs
        return Language(tsjs.language())


def _node_text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_jsdoc(node: Node, src: bytes) -> Optional[str]:
    """Look for a leading block comment (JSDoc) immediately before the node."""
    # Tree-sitter doesn't expose siblings easily; skip for now and return None.
    return None


class JavaScriptParser(BaseParser):
    """Parses JS/TS files using Tree-sitter."""

    def __init__(self) -> None:
        # Initialise with JS; will swap to TS on first .ts parse
        import tree_sitter_javascript as tsjs
        self._js_lang = Language(tsjs.language())
        try:
            import tree_sitter_typescript as tsts
            self._ts_lang = Language(tsts.language_typescript())
        except Exception:
            self._ts_lang = self._js_lang
        self._parsers: dict[str, Parser] = {}

    def _get_parser(self, file_path: str) -> Parser:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self._parsers:
            lang = self._ts_lang if ext == ".ts" else self._js_lang
            self._parsers[ext] = Parser(lang)
        return self._parsers[ext]

    # ──────────────────────────────────────────────────────────────────────────
    def parse(self, source: bytes, file_path: str) -> tuple[list[NodeData], list[EdgeData]]:
        parser = self._get_parser(file_path)
        tree = parser.parse(source)
        root = tree.root_node

        nodes: list[NodeData] = []
        edges: list[EdgeData] = []

        lang_label = "typescript" if file_path.endswith(".ts") else "javascript"
        file_node = NodeData(
            id=file_path,
            label="File",
            properties={"path": file_path, "name": os.path.basename(file_path), "language": lang_label},
            embed_text=f"{lang_label.capitalize()} file: {file_path}",
        )
        nodes.append(file_node)
        self._walk(root, source, file_path, lang_label, nodes, edges)
        return nodes, edges

    # ──────────────────────────────────────────────────────────────────────────
    def _walk(self, root: Node, src: bytes, file_path: str, lang: str, nodes: list, edges: list) -> None:
        for child in root.children:
            t = child.type
            if t in ("import_statement", "import_declaration"):
                self._handle_import(child, src, file_path, nodes, edges)
            elif t in ("class_declaration", "class_expression"):
                self._handle_class(child, src, file_path, lang, nodes, edges)
            elif t in ("function_declaration", "function_expression", "arrow_function"):
                self._handle_function(child, src, file_path, lang, nodes, edges)
            elif t == "lexical_declaration":
                # const foo = () => ...  or  const foo = function() {...}
                for decl in child.children:
                    if decl.type == "variable_declarator":
                        val = decl.child_by_field_name("value")
                        if val and val.type in ("arrow_function", "function_expression"):
                            self._handle_function(val, src, file_path, lang, nodes, edges,
                                                  name_override=_node_text(decl.child_by_field_name("name") or decl, src))
            elif t == "interface_declaration":
                self._handle_interface(child, src, file_path, lang, nodes, edges)
            elif t == "type_alias_declaration":
                self._handle_type_alias(child, src, file_path, lang, nodes, edges)
            elif t == "export_statement":
                # Recurse one level into exports
                for inner in child.children:
                    if inner.type in ("class_declaration", "function_declaration",
                                     "interface_declaration", "type_alias_declaration"):
                        self._walk_single(inner, src, file_path, lang, nodes, edges)

    def _walk_single(self, node: Node, src: bytes, file_path: str, lang: str, nodes: list, edges: list) -> None:
        t = node.type
        if t in ("class_declaration", "class_expression"):
            self._handle_class(node, src, file_path, lang, nodes, edges)
        elif t in ("function_declaration", "function_expression", "arrow_function"):
            self._handle_function(node, src, file_path, lang, nodes, edges)
        elif t == "interface_declaration":
            self._handle_interface(node, src, file_path, lang, nodes, edges)
        elif t == "type_alias_declaration":
            self._handle_type_alias(node, src, file_path, lang, nodes, edges)

    # ── Import ────────────────────────────────────────────────────────────────
    def _handle_import(self, node: Node, src: bytes, file_path: str, nodes: list, edges: list) -> None:
        # Look for the string source
        for child in node.children:
            if child.type == "string":
                raw = _node_text(child, src).strip("'\"` ")
                mod_id = f"module::{raw}"
                if not any(n.id == mod_id for n in nodes):
                    nodes.append(NodeData(id=mod_id, label="Module", properties={"name": raw}))
                edges.append(EdgeData(type="IMPORTS", source_id=file_path, target_id=mod_id))
                break

    # ── Class ─────────────────────────────────────────────────────────────────
    def _handle_class(self, node: Node, src: bytes, file_path: str, lang: str, nodes: list, edges: list) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        class_name = _node_text(name_node, src)
        class_id = f"{file_path}::{class_name}"

        nodes.append(NodeData(
            id=class_id, label="Class",
            properties={"name": class_name, "file_path": file_path,
                        "lineno": node.start_point[0] + 1, "docstring": "", "language": lang},
            embed_text=f"Class {class_name}",
        ))
        edges.append(EdgeData(type="DEFINES", source_id=file_path, target_id=class_id))

        # Inheritance — tree-sitter-javascript 0.25 uses a `class_heritage` child,
        # not a `superclass` field.
        heritage = None
        for child in node.children:
            if child.type == "class_heritage":
                heritage = child
                break
        if heritage:
            for hc in heritage.children:
                if hc.type == "identifier":
                    parent_name = _node_text(hc, src)
                    parent_id = f"{file_path}::{parent_name}"
                    edges.append(EdgeData(type="INHERITS", source_id=class_id, target_id=parent_id))
                    break

        # Methods inside class body
        body = node.child_by_field_name("body")
        if body:
            for item in body.children:
                if item.type == "method_definition":
                    self._handle_method(item, src, file_path, class_id, class_name, lang, nodes, edges)

    # ── Function ──────────────────────────────────────────────────────────────
    def _handle_function(self, node: Node, src: bytes, file_path: str, lang: str,
                         nodes: list, edges: list, name_override: str | None = None) -> None:
        name_node = node.child_by_field_name("name")
        fn_name = name_override or ((_node_text(name_node, src)) if name_node else "<anonymous>")
        fn_id = f"{file_path}::{fn_name}"

        params_node = node.child_by_field_name("parameters") or node.child_by_field_name("parameter")
        sig = f"function {fn_name}({_node_text(params_node, src) if params_node else ''})"

        nodes.append(NodeData(
            id=fn_id, label="Function",
            properties={"name": fn_name, "file_path": file_path,
                        "lineno": node.start_point[0] + 1, "docstring": "", "signature": sig, "language": lang},
            embed_text=f"Function {sig}",
        ))
        edges.append(EdgeData(type="DEFINES", source_id=file_path, target_id=fn_id))

    # ── Method ────────────────────────────────────────────────────────────────
    def _handle_method(self, node: Node, src: bytes, file_path: str, class_id: str,
                       class_name: str, lang: str, nodes: list, edges: list) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        method_name = _node_text(name_node, src)
        method_id = f"{class_id}::{method_name}"

        nodes.append(NodeData(
            id=method_id, label="Method",
            properties={"name": method_name, "parent_name": class_name, "file_path": file_path,
                        "lineno": node.start_point[0] + 1, "docstring": "", "signature": method_name, "language": lang},
            embed_text=f"Method {class_name}.{method_name}",
        ))
        edges.append(EdgeData(type="HAS_METHOD", source_id=class_id, target_id=method_id))

    # ── Interface ─────────────────────────────────────────────────────────────
    def _handle_interface(self, node: Node, src: bytes, file_path: str, lang: str, nodes: list, edges: list) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        iface_name = _node_text(name_node, src)
        iface_id = f"{file_path}::{iface_name}"

        nodes.append(NodeData(
            id=iface_id, label="Interface",
            properties={"name": iface_name, "file_path": file_path,
                        "lineno": node.start_point[0] + 1, "docstring": "", "language": lang},
            embed_text=f"Interface {iface_name}",
        ))
        edges.append(EdgeData(type="DEFINES", source_id=file_path, target_id=iface_id))

    # ── TypeAlias ─────────────────────────────────────────────────────────────
    def _handle_type_alias(self, node: Node, src: bytes, file_path: str, lang: str, nodes: list, edges: list) -> None:
        name_node = node.child_by_field_name("name")
        type_node = node.child_by_field_name("value") or node.child_by_field_name("type")
        if not name_node:
            return
        alias_name = _node_text(name_node, src)
        alias_id = f"{file_path}::{alias_name}"
        target_type = _node_text(type_node, src) if type_node else ""

        nodes.append(NodeData(
            id=alias_id, label="TypeAlias",
            properties={"name": alias_name, "file_path": file_path,
                        "lineno": node.start_point[0] + 1, "target_type": target_type, "language": lang},
            embed_text=f"TypeAlias {alias_name} = {target_type}",
        ))
        edges.append(EdgeData(type="DEFINES", source_id=file_path, target_id=alias_id))
