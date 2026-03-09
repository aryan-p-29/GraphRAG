"""
tests/test_js_parser.py
────────────────────────
Unit tests for JavaScriptParser (JS and TS).
"""

import pytest
from parsers.javascript_parser import JavaScriptParser

JS_SAMPLE = b'''
import { readFile } from 'fs';
import axios from 'axios';

class EventEmitter {
    constructor() {
        this.listeners = {};
    }
    on(event, callback) {}
    emit(event, data) {}
}

class MyEmitter extends EventEmitter {
    fire() {}
}

function debounce(fn, wait) {
    let timer;
    return function() {};
}

const throttle = (fn, limit) => {
    return () => {};
};
'''

TS_SAMPLE = b'''
import { Component } from '@angular/core';

interface Repository {
    name: string;
    clone(): void;
}

type UUID = string;

class GitHubRepo implements Repository {
    name: string;
    constructor(name: string) {
        this.name = name;
    }
    clone(): void {}
}
'''

@pytest.fixture
def js_parsed():
    p = JavaScriptParser()
    return p.parse(JS_SAMPLE, "src/events.js")

@pytest.fixture
def ts_parsed():
    p = JavaScriptParser()
    return p.parse(TS_SAMPLE, "src/repo.ts")


def node_by_label(nodes, label):
    return [n for n in nodes if n.label == label]

def edge_by_type(edges, t):
    return [e for e in edges if e.type == t]


# ── JavaScript tests ───────────────────────────────────────────────────────────

def test_js_file_node(js_parsed):
    nodes, _ = js_parsed
    f = node_by_label(nodes, "File")
    assert len(f) == 1
    assert f[0].properties["language"] == "javascript"

def test_js_classes(js_parsed):
    nodes, _ = js_parsed
    names = {n.properties["name"] for n in node_by_label(nodes, "Class")}
    assert "EventEmitter" in names
    assert "MyEmitter" in names

def test_js_functions(js_parsed):
    nodes, _ = js_parsed
    names = {n.properties["name"] for n in node_by_label(nodes, "Function")}
    assert "debounce" in names or "throttle" in names

def test_js_methods(js_parsed):
    nodes, _ = js_parsed
    names = {n.properties["name"] for n in node_by_label(nodes, "Method")}
    assert "on" in names or "emit" in names

def test_js_imports(js_parsed):
    nodes, edges = js_parsed
    mods = {n.properties["name"] for n in node_by_label(nodes, "Module")}
    assert "fs" in mods or "axios" in mods

def test_js_inherits(js_parsed):
    _, edges = js_parsed
    inh = edge_by_type(edges, "INHERITS")
    assert any("MyEmitter" in e.source_id for e in inh)


# ── TypeScript tests ───────────────────────────────────────────────────────────

def test_ts_file_node(ts_parsed):
    nodes, _ = ts_parsed
    f = node_by_label(nodes, "File")
    assert f[0].properties["language"] == "typescript"

def test_ts_interface(ts_parsed):
    nodes, _ = ts_parsed
    ifaces = node_by_label(nodes, "Interface")
    names = {n.properties["name"] for n in ifaces}
    assert "Repository" in names

def test_ts_type_alias(ts_parsed):
    nodes, _ = ts_parsed
    aliases = node_by_label(nodes, "TypeAlias")
    names = {n.properties["name"] for n in aliases}
    assert "UUID" in names

def test_ts_class(ts_parsed):
    nodes, _ = ts_parsed
    classes = node_by_label(nodes, "Class")
    names = {n.properties["name"] for n in classes}
    assert "GitHubRepo" in names

def test_ts_defines_edges(ts_parsed):
    _, edges = ts_parsed
    defines = edge_by_type(edges, "DEFINES")
    targets = {e.target_id for e in defines}
    assert any("Repository" in t for t in targets)
    assert any("UUID" in t for t in targets)
