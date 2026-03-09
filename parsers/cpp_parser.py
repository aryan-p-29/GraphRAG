"""
parsers/cpp_parser.py
──────────────────────
Tree-sitter-based parser for C++ source files (.cpp, .hpp, .cc, .h).

Extracts:
  Nodes : File, Class, Struct, Function, Method, TypeAlias, Module (namespace)
  Edges : DEFINES, HAS_METHOD, CALLS, IMPORTS, INHERITS
"""

from __future__ import annotations

import os
from typing import Optional

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser, Node

from graph_schema import NodeData, EdgeData
from parsers import BaseParser

CPP_LANGUAGE = Language(tscpp.language())


def _node_text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _first_child_of_type(node: Node, *types: str) -> Optional[Node]:
    for child in node.children:
        if child.type in types:
            return child
    return None


class CppParser(BaseParser):
    """Parses C++ files using Tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(CPP_LANGUAGE)

    # ──────────────────────────────────────────────────────────────────────────
    def parse(self, source: bytes, file_path: str) -> tuple[list[NodeData], list[EdgeData]]:
        tree = self._parser.parse(source)
        root = tree.root_node

        nodes: list[NodeData] = []
        edges: list[EdgeData] = []

        nodes.append(NodeData(
            id=file_path, label="File",
            properties={"path": file_path, "name": os.path.basename(file_path), "language": "cpp"},
            embed_text=f"C++ file: {file_path}",
        ))

        self._walk(root, source, file_path, None, nodes, edges)
        return nodes, edges

    # ──────────────────────────────────────────────────────────────────────────
    def _walk(
        self,
        node: Node,
        src: bytes,
        file_path: str,
        parent_id: Optional[str],
        nodes: list,
        edges: list,
    ) -> None:
        scope_id = parent_id or file_path

        for child in node.children:
            t = child.type

            if t == "preproc_include":
                self._handle_include(child, src, file_path, nodes, edges)

            elif t == "namespace_definition":
                self._handle_namespace(child, src, file_path, nodes, edges)

            elif t in ("class_specifier", "struct_specifier"):
                self._handle_class_or_struct(child, src, file_path, scope_id, nodes, edges)

            elif t == "function_definition":
                self._handle_function_def(child, src, file_path, scope_id, nodes, edges)

            elif t == "declaration":
                # May contain function pointer / inline declarations; skip for now
                pass

            elif t in ("using_declaration", "type_alias_declaration", "alias_declaration"):
                self._handle_type_alias(child, src, file_path, nodes, edges)

            elif t == "template_declaration":
                # Recurse into the actual declaration inside the template
                inner = _first_child_of_type(child, "class_specifier", "struct_specifier", "function_definition")
                if inner:
                    self._walk(inner, src, file_path, scope_id, nodes, edges)

    # ── #include ──────────────────────────────────────────────────────────────
    def _handle_include(self, node: Node, src: bytes, file_path: str, nodes: list, edges: list) -> None:
        for child in node.children:
            if child.type in ("string_literal", "system_lib_string"):
                raw = _node_text(child, src).strip('<>"')
                name = os.path.basename(raw).replace(".h", "").replace(".hpp", "")
                mod_id = f"module::{name}"
                if not any(n.id == mod_id for n in nodes):
                    nodes.append(NodeData(id=mod_id, label="Module",
                                          properties={"name": name, "header": raw}))
                edges.append(EdgeData(type="IMPORTS", source_id=file_path, target_id=mod_id))

    # ── namespace → Module ────────────────────────────────────────────────────
    def _handle_namespace(self, node: Node, src: bytes, file_path: str, nodes: list, edges: list) -> None:
        name_node = node.child_by_field_name("name")
        ns_name = _node_text(name_node, src) if name_node else "<anonymous>"
        ns_id = f"module::{ns_name}"
        if not any(n.id == ns_id for n in nodes):
            nodes.append(NodeData(
                id=ns_id, label="Module",
                properties={"name": ns_name, "kind": "namespace"},
                embed_text=f"C++ namespace {ns_name}",
            ))
        edges.append(EdgeData(type="IMPORTS", source_id=file_path, target_id=ns_id))

        # Recurse into the namespace body
        body = node.child_by_field_name("body")
        if body:
            self._walk(body, src, file_path, ns_id, nodes, edges)

    # ── class / struct ────────────────────────────────────────────────────────
    def _handle_class_or_struct(
        self, node: Node, src: bytes, file_path: str,
        parent_id: str, nodes: list, edges: list
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        type_name = _node_text(name_node, src)
        label = "Class" if node.type == "class_specifier" else "Struct"
        type_id = f"{file_path}::{type_name}"

        nodes.append(NodeData(
            id=type_id, label=label,
            properties={"name": type_name, "file_path": file_path,
                        "lineno": node.start_point[0] + 1, "docstring": "", "language": "cpp"},
            embed_text=f"C++ {label.lower()} {type_name}",
        ))
        edges.append(EdgeData(type="DEFINES", source_id=file_path, target_id=type_id))

        # Base classes (inheritance)
        base_clause = _first_child_of_type(node, "base_class_clause")
        if base_clause:
            for base in base_clause.children:
                if base.type == "type_identifier":
                    base_id = f"{file_path}::{_node_text(base, src)}"
                    edges.append(EdgeData(type="INHERITS", source_id=type_id, target_id=base_id))

        # Methods inside the body
        body = _first_child_of_type(node, "field_declaration_list")
        if body:
            for item in body.children:
                if item.type == "function_definition":
                    self._handle_method_def(item, src, file_path, type_id, type_name, nodes, edges)
                elif item.type == "declaration":
                    # e.g. `void foo();` (declaration without body)
                    # We create a Method stub
                    self._handle_declaration_in_class(item, src, file_path, type_id, type_name, nodes, edges)

    # ── Free function / out-of-class method definition ─────────────────────────
    def _handle_function_def(
        self, node: Node, src: bytes, file_path: str,
        parent_id: str, nodes: list, edges: list
    ) -> None:
        declarator = node.child_by_field_name("declarator")
        if not declarator:
            return

        # Qualified method: MyClass::method_name
        if declarator.type == "function_declarator":
            name_part = declarator.child_by_field_name("declarator")
            if name_part and name_part.type == "qualified_identifier":
                # scope + name
                scope_node = name_part.child_by_field_name("scope")
                inner_name_node = name_part.child_by_field_name("name")
                if scope_node and inner_name_node:
                    class_name = _node_text(scope_node, src).rstrip(":")
                    method_name = _node_text(inner_name_node, src)
                    class_id = f"{file_path}::{class_name}"
                    method_id = f"{class_id}::{method_name}"
                    nodes.append(NodeData(
                        id=method_id, label="Method",
                        properties={"name": method_name, "parent_name": class_name, "file_path": file_path,
                                    "lineno": node.start_point[0] + 1, "docstring": "", "signature": method_name, "language": "cpp"},
                        embed_text=f"C++ method {class_name}::{method_name}",
                    ))
                    edges.append(EdgeData(type="HAS_METHOD", source_id=class_id, target_id=method_id))
                    body = node.child_by_field_name("body")
                    if body:
                        self._extract_calls(body, src, method_id, file_path, edges)
                    return

            # Plain function
            fn_name_node = declarator.child_by_field_name("declarator")
            fn_name = _node_text(fn_name_node, src) if fn_name_node else "<unknown>"
        else:
            fn_name = _node_text(declarator, src)

        fn_id = f"{file_path}::{fn_name}"
        nodes.append(NodeData(
            id=fn_id, label="Function",
            properties={"name": fn_name, "file_path": file_path,
                        "lineno": node.start_point[0] + 1, "docstring": "", "signature": fn_name, "language": "cpp"},
            embed_text=f"C++ function {fn_name}",
        ))
        edges.append(EdgeData(type="DEFINES", source_id=file_path, target_id=fn_id))
        body = node.child_by_field_name("body")
        if body:
            self._extract_calls(body, src, fn_id, file_path, edges)

    # ── Method inside class body ──────────────────────────────────────────────
    def _handle_method_def(
        self, node: Node, src: bytes, file_path: str,
        class_id: str, class_name: str, nodes: list, edges: list
    ) -> None:
        declarator = node.child_by_field_name("declarator")
        if not declarator:
            return
        fn_decl = declarator if declarator.type == "function_declarator" \
            else _first_child_of_type(declarator, "function_declarator")
        if not fn_decl:
            return
        name_part = fn_decl.child_by_field_name("declarator")
        if not name_part:
            return
        method_name = _node_text(name_part, src)
        method_id = f"{class_id}::{method_name}"

        nodes.append(NodeData(
            id=method_id, label="Method",
            properties={"name": method_name, "parent_name": class_name, "file_path": file_path,
                        "lineno": node.start_point[0] + 1, "docstring": "", "signature": method_name, "language": "cpp"},
            embed_text=f"C++ method {class_name}::{method_name}",
        ))
        edges.append(EdgeData(type="HAS_METHOD", source_id=class_id, target_id=method_id))
        body = node.child_by_field_name("body")
        if body:
            self._extract_calls(body, src, method_id, file_path, edges)

    def _handle_declaration_in_class(
        self, node: Node, src: bytes, file_path: str,
        class_id: str, class_name: str, nodes: list, edges: list
    ) -> None:
        """Handle forward-declared member functions inside a class body."""
        for child in node.children:
            if child.type == "function_declarator":
                name_part = child.child_by_field_name("declarator")
                if name_part:
                    method_name = _node_text(name_part, src)
                    method_id = f"{class_id}::{method_name}"
                    if not any(n.id == method_id for n in nodes):
                        nodes.append(NodeData(
                            id=method_id, label="Method",
                            properties={"name": method_name, "parent_name": class_name, "file_path": file_path,
                                        "lineno": node.start_point[0] + 1, "docstring": "", "signature": method_name, "language": "cpp"},
                            embed_text=f"C++ method {class_name}::{method_name}",
                        ))
                        edges.append(EdgeData(type="HAS_METHOD", source_id=class_id, target_id=method_id))

    # ── using X = Y / typedef ─────────────────────────────────────────────────
    def _handle_type_alias(self, node: Node, src: bytes, file_path: str, nodes: list, edges: list) -> None:
        # tree-sitter-cpp 0.23 uses `alias_declaration` for `using X = Y;`
        # `using_declaration` is a namespace inject (e.g. `using std::cout;`) — skip.
        if node.type not in ("alias_declaration", "type_alias_declaration"):
            return
        # Find the name (type_identifier) and value (type_descriptor) children
        name_node = None
        type_node = None
        for child in node.named_children:
            if child.type == "type_identifier" and name_node is None:
                name_node = child
            elif child.type in ("type_descriptor", "type_identifier") and name_node is not None:
                type_node = child
        if not name_node:
            return
        alias_name = _node_text(name_node, src)
        alias_id = f"{file_path}::{alias_name}"
        target = _node_text(type_node, src) if type_node else ""
        nodes.append(NodeData(
            id=alias_id, label="TypeAlias",
            properties={"name": alias_name, "file_path": file_path,
                        "lineno": node.start_point[0] + 1, "target_type": target, "language": "cpp"},
            embed_text=f"C++ type alias {alias_name} = {target}",
        ))
        edges.append(EdgeData(type="DEFINES", source_id=file_path, target_id=alias_id))

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
                    raw = _node_text(fn_node, src)
                    # Strip scope qualifiers: std::cout → cout, obj.method → method
                    callee = raw.split("::")[-1].split(".")[-1].split("->")[-1]
                    callee_id = f"{file_path}::{callee}"
                    edges.append(EdgeData(type="CALLS", source_id=caller_id, target_id=callee_id))
            for child in n.children:
                recurse(child)

        recurse(body)
