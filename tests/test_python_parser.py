"""
tests/test_python_parser.py
────────────────────────────
Unit tests for PythonParser.
"""

import pytest
from parsers.python_parser import PythonParser

SAMPLE = b'''
import os
from pathlib import Path

class Animal:
    """Base animal class."""
    def __init__(self, name: str):
        """Initialize the animal."""
        self.name = name

    def speak(self) -> str:
        """Make a sound."""
        return ""

class Dog(Animal):
    """Dog subclass."""
    def speak(self) -> str:
        """Bark."""
        return "Woof"

def helper(x: int) -> int:
    """Double x."""
    return x * 2
'''

@pytest.fixture
def parsed():
    parser = PythonParser()
    nodes, edges = parser.parse(SAMPLE, "test/sample.py")
    return nodes, edges


def node_by_label(nodes, label):
    return [n for n in nodes if n.label == label]


def edge_by_type(edges, etype):
    return [e for e in edges if e.type == etype]


def test_file_node(parsed):
    nodes, _ = parsed
    files = node_by_label(nodes, "File")
    assert len(files) == 1
    assert files[0].id == "test/sample.py"
    assert files[0].properties["language"] == "python"


def test_classes_extracted(parsed):
    nodes, _ = parsed
    classes = node_by_label(nodes, "Class")
    names = {n.properties["name"] for n in classes}
    assert "Animal" in names
    assert "Dog" in names


def test_function_extracted(parsed):
    nodes, _ = parsed
    funcs = node_by_label(nodes, "Function")
    names = {n.properties["name"] for n in funcs}
    assert "helper" in names


def test_methods_extracted(parsed):
    nodes, _ = parsed
    methods = node_by_label(nodes, "Method")
    names = {n.properties["name"] for n in methods}
    assert "__init__" in names
    assert "speak" in names


def test_imports_extracted(parsed):
    nodes, edges = parsed
    modules = node_by_label(nodes, "Module")
    mod_names = {n.properties["name"] for n in modules}
    assert "os" in mod_names
    assert "pathlib" in mod_names
    import_edges = edge_by_type(edges, "IMPORTS")
    assert len(import_edges) >= 2


def test_defines_edges(parsed):
    _, edges = parsed
    defines = edge_by_type(edges, "DEFINES")
    targets = {e.target_id for e in defines}
    assert "test/sample.py::Animal" in targets
    assert "test/sample.py::Dog" in targets
    assert "test/sample.py::helper" in targets


def test_has_method_edges(parsed):
    _, edges = parsed
    hm = edge_by_type(edges, "HAS_METHOD")
    assert len(hm) >= 3  # __init__, speak x2


def test_inherits_edge(parsed):
    _, edges = parsed
    inh = edge_by_type(edges, "INHERITS")
    assert any(e.source_id == "test/sample.py::Dog" for e in inh)


def test_docstrings_captured(parsed):
    nodes, _ = parsed
    animal = next(n for n in nodes if n.properties.get("name") == "Animal")
    assert "animal" in animal.properties["docstring"].lower()


def test_embed_text_set(parsed):
    nodes, _ = parsed
    for n in nodes:
        if n.label in ("Class", "Function", "Method"):
            assert n.embed_text is not None and len(n.embed_text) > 0
