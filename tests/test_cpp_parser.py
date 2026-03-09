"""
tests/test_cpp_parser.py
─────────────────────────
Unit tests for CppParser.
"""

import pytest
from parsers.cpp_parser import CppParser

SAMPLE = b'''
#include <iostream>
#include <string>

namespace animals {

class Animal {
public:
    Animal(std::string name) : name_(name) {}
    virtual std::string speak() = 0;

protected:
    std::string name_;
};

struct Point {
    double x;
    double y;
    double distance();
};

class Dog : public Animal {
public:
    Dog(std::string name) : Animal(name) {}
    std::string speak() override { return "Woof"; }
    void fetch(std::string item);
};

} // namespace animals

using Weight = double;

double animals::Point::distance() {
    return 0.0;
}

void animals::Dog::fetch(std::string item) {
    std::cout << item << std::endl;
}

int main() {
    animals::Dog d("Rex");
    d.speak();
    return 0;
}
'''

@pytest.fixture
def parsed():
    p = CppParser()
    return p.parse(SAMPLE, "src/animals.cpp")

def node_by_label(nodes, label):
    return [n for n in nodes if n.label == label]

def edge_by_type(edges, t):
    return [e for e in edges if e.type == t]


def test_file_node(parsed):
    nodes, _ = parsed
    f = node_by_label(nodes, "File")
    assert len(f) == 1
    assert f[0].properties["language"] == "cpp"

def test_classes_extracted(parsed):
    nodes, _ = parsed
    classes = node_by_label(nodes, "Class")
    names = {n.properties["name"] for n in classes}
    assert "Animal" in names
    assert "Dog" in names

def test_struct_extracted(parsed):
    nodes, _ = parsed
    structs = node_by_label(nodes, "Struct")
    names = {n.properties["name"] for n in structs}
    assert "Point" in names

def test_namespace_as_module(parsed):
    nodes, _ = parsed
    modules = node_by_label(nodes, "Module")
    names = {n.properties["name"] for n in modules}
    assert "animals" in names

def test_includes_as_imports(parsed):
    nodes, edges = parsed
    import_edges = edge_by_type(edges, "IMPORTS")
    assert len(import_edges) >= 1  # iostream, string, animals namespace

def test_type_alias(parsed):
    nodes, _ = parsed
    aliases = node_by_label(nodes, "TypeAlias")
    names = {n.properties["name"] for n in aliases}
    assert "Weight" in names

def test_inherits_edge(parsed):
    _, edges = parsed
    inh = edge_by_type(edges, "INHERITS")
    assert any("Dog" in e.source_id for e in inh)

def test_methods_in_class(parsed):
    _, edges = parsed
    hm = edge_by_type(edges, "HAS_METHOD")
    assert len(hm) >= 1

def test_defines_edges(parsed):
    _, edges = parsed
    defines = edge_by_type(edges, "DEFINES")
    targets = {e.target_id for e in defines}
    assert any("Animal" in t for t in targets)
    assert any("Dog" in t for t in targets)
    assert any("Point" in t for t in targets)
    assert any("Weight" in t for t in targets)
