"""
House style for anything the user reads.

Dashes are the rule worth enforcing mechanically, because they creep back in
one caption at a time and nobody notices until the interface is full of them.
Docstrings and comments are exempt: they are for whoever is reading the source,
not for the person using the app.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tabled import config

APP = config.ROOT / "app.py"
SERVE = config.ROOT / "src" / "tabled" / "serve"

BANNED = {
    "—": "em dash",
    "–": "en dash",
    "‒": "figure dash",
    "−": "minus sign",
}

# Literal shell commands and filenames are instructions, not prose: the flag
# in `tabled fit --model item-item` has to survive intact to be typed.
EXEMPT = ("--model", "--holdout", "tabled-session.json", "item-item")


def user_facing_strings(path: Path) -> list[tuple[int, str]]:
    """Every string constant that is not a docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"))

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))

    return [(node.lineno, node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings]


@pytest.mark.parametrize("path", [APP], ids=lambda p: p.name)
def test_no_dashes_in_anything_the_user_reads(path):
    offences = []
    for line, text in user_facing_strings(path):
        if any(token in text for token in EXEMPT):
            continue
        for char, name in BANNED.items():
            if char in text:
                offences.append(f"{path.name}:{line} {name} in {text[:70]!r}")

    assert not offences, "\n".join(offences)


def test_the_guard_would_actually_catch_one(tmp_path):
    """A test that never fails is not a test."""
    sample = tmp_path / "sample.py"
    sample.write_text('"""Docstring — allowed."""\nx = "Caption — banned"\n',
                      encoding="utf-8")

    found = [text for _, text in user_facing_strings(sample)]
    assert found == ["Caption — banned"]
    assert any("—" in text for text in found)
