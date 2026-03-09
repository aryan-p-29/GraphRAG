"""
tests/test_go_parser.py
────────────────────────
Unit tests for GoParser.
"""

import pytest
from parsers.go_parser import GoParser

SAMPLE = b'''
package main

import (
    "fmt"
    "github.com/some/lib"
)

type Animal interface {
    Speak() string
}

type Dog struct {
    Name string
    Age  int
}

func (d Dog) Speak() string {
    return "Woof"
}

func (d Dog) Fetch(item string) {
    fmt.Println(d.Name, "fetches", item)
}

func NewDog(name string) Dog {
    return Dog{Name: name}
}
'''

@pytest.fixture
def parsed():
    p = GoParser()
    return p.parse(SAMPLE, "main/dog.go")

def node_by_label(nodes, label):
    return [n for n in nodes if n.label == label]

def edge_by_type(edges, t):
    return [e for e in edges if e.type == t]


def test_file_node(parsed):
    nodes, _ = parsed
    f = node_by_label(nodes, "File")
    assert len(f) == 1
    assert f[0].properties["language"] == "go"

def test_struct_extracted(parsed):
    nodes, _ = parsed
    structs = node_by_label(nodes, "Struct")
    names = {n.properties["name"] for n in structs}
    assert "Dog" in names

def test_interface_extracted(parsed):
    nodes, _ = parsed
    ifaces = node_by_label(nodes, "Interface")
    names = {n.properties["name"] for n in ifaces}
    assert "Animal" in names

def test_function_extracted(parsed):
    nodes, _ = parsed
    funcs = node_by_label(nodes, "Function")
    names = {n.properties["name"] for n in funcs}
    assert "NewDog" in names

def test_methods_extracted(parsed):
    nodes, _ = parsed
    methods = node_by_label(nodes, "Method")
    names = {n.properties["name"] for n in methods}
    assert "Speak" in names
    assert "Fetch" in names

def test_imports_extracted(parsed):
    nodes, edges = parsed
    mods = node_by_label(nodes, "Module")
    mod_names = {n.properties["name"] for n in mods}
    assert "fmt" in mod_names or "lib" in mod_names
    assert len(edge_by_type(edges, "IMPORTS")) >= 1

def test_defines_edges(parsed):
    _, edges = parsed
    defines = edge_by_type(edges, "DEFINES")
    targets = {e.target_id for e in defines}
    assert any("Dog" in t for t in targets)
    assert any("Animal" in t for t in targets)
    assert any("NewDog" in t for t in targets)

def test_has_method_edges(parsed):
    _, edges = parsed
    hm = edge_by_type(edges, "HAS_METHOD")
    assert len(hm) >= 2

def test_method_parent_name(parsed):
    nodes, _ = parsed
    methods = node_by_label(nodes, "Method")
    for m in methods:
        assert m.properties["parent_name"] == "Dog"
