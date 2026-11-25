"""Instantiate pluggable components (systems, judges) from a builtin alias or import path."""

from __future__ import annotations

import importlib
from collections.abc import Mapping


def load_plugin(spec: str, builtins: Mapping[str, str], kind: str) -> object:
    """Resolve ``spec`` (a key of ``builtins`` or ``package.module:ClassName``) and call it
    with no arguments. Callers check the result against their protocol.
    """
    target = builtins.get(spec, spec)
    module_name, sep, attr = target.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(
            f"invalid {kind} spec {spec!r}: expected a builtin "
            f"({', '.join(sorted(builtins))}) or 'module:ClassName'"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    instance: object = factory()
    return instance
