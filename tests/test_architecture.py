"""
Enforce the layering.

A directory structure only survives if something checks it. The rule is a
strict order — each layer may import from layers below it, never from layers
above or beside it:

    config          (imports nothing internal)
      -> data
        -> models
          -> eval
            -> cli  (the only module allowed to see everything)

The one that actually bites is `eval` and `models`: it would be very easy for
a model to reach into the harness for a metric, and then the thing being
measured would depend on the thing measuring it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import tabled

PACKAGE = Path(tabled.__file__).parent

# `eval` and `serve` sit at the same level deliberately: they are siblings,
# so neither may import the other. Scoring a model and serving a card are
# separate concerns, and coupling them would make the harness depend on UI
# decisions like the popularity gate.
LAYERS = {"config": 0, "data": 1, "models": 2, "eval": 3, "serve": 3,
          "cli": 4}


def layer_of(path: Path) -> tuple[str, int] | None:
    top = path.relative_to(PACKAGE).parts[0]
    name = top[:-3] if top.endswith(".py") else top
    return (name, LAYERS[name]) if name in LAYERS else None


def internal_imports(path: Path) -> set[str]:
    """Top-level `tabled` subpackages imported by this file, either form."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:                       # relative import
                continue
            module = node.module or ""
            if module == "tabled":
                found.update(a.name for a in node.names if a.name in LAYERS)
            elif module.startswith("tabled."):
                found.add(module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("tabled."):
                    found.add(alias.name.split(".")[1])
    return {f for f in found if f in LAYERS}


def python_files() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.stem)
def test_no_upward_or_sideways_imports(path):
    info = layer_of(path)
    if info is None:
        pytest.skip(f"{path} is not in a known layer")
    name, level = info

    for imported in internal_imports(path):
        if imported == name:
            continue                             # within a package is fine
        assert LAYERS[imported] < level, (
            f"{path.relative_to(PACKAGE)} (layer '{name}') imports "
            f"'{imported}' (layer {LAYERS[imported]}). "
            f"Only strictly lower layers are allowed.")


def test_config_imports_nothing_internal():
    assert internal_imports(PACKAGE / "config.py") == set()


def test_models_do_not_depend_on_the_harness_that_scores_them():
    for path in (PACKAGE / "models").rglob("*.py"):
        assert "eval" not in internal_imports(path), (
            f"{path.name} makes the thing being measured depend on the thing "
            f"measuring it")


def test_every_subpackage_has_an_init():
    for d in PACKAGE.rglob("*"):
        if d.is_dir() and d.name != "__pycache__" and any(d.glob("*.py")):
            assert (d / "__init__.py").exists(), f"{d} is missing __init__.py"


def test_config_path_constants_are_all_used():
    """An unused path constant drifts out of sync with what the code writes."""
    import tabled.config as cfg

    config_src = (PACKAGE / "config.py").read_text(encoding="utf-8")
    # app.py lives outside the package but is still project code that reads
    # config, so it counts as a use.
    sources = [*PACKAGE.rglob("*.py"), *Path(__file__).parent.rglob("*.py"),
               cfg.ROOT / "app.py"]
    elsewhere = "\n".join(p.read_text(encoding="utf-8") for p in sources
                          if p.name != "config.py" and p.exists())

    names = [n for n in dir(cfg)
             if n.isupper() and isinstance(getattr(cfg, n), Path)]
    assert names, "no path constants found — did config.py move?"

    for name in names:
        derives_others = config_src.count(name) > 1
        assert name in elsewhere or derives_others, (
            f"config.{name} is defined but never used")
