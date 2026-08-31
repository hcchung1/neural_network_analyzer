"""Backward-compatible facade for the split configuration and data modules."""

from __future__ import annotations

import os
import sys
import types

_TRANSFORMER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TRANSFORMER_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config as _config
from data import features as _features
from data import loading as _loading
from data import schema as _schema

_TARGET_MODULES = (_config, _schema, _features, _loading)
for _module in _TARGET_MODULES:
    for _name in getattr(_module, "__all__", ()):
        globals()[_name] = getattr(_module, _name)

OPTIONS_SOURCE_PATH = os.path.join(_TRANSFORMER_DIR, "config.py")


def configure_runtime_sequence_length(raw_max_seq_len: int):
    """Synchronize the cached and physical sequence lengths across data modules."""
    raw = max(1, int(raw_max_seq_len))
    _schema.NEW_FORMAT_MAX_SEQ_LEN = raw
    resolved = _schema.resolve_runtime_schema(raw)
    for target in _TARGET_MODULES:
        if hasattr(target, "NEW_FORMAT_MAX_SEQ_LEN"):
            setattr(target, "NEW_FORMAT_MAX_SEQ_LEN", raw)
        if hasattr(target, "MAX_SEQ_LEN_FALLBACK"):
            setattr(target, "MAX_SEQ_LEN_FALLBACK", resolved.physical_seq_len)
    globals()["NEW_FORMAT_MAX_SEQ_LEN"] = raw
    globals()["MAX_SEQ_LEN_FALLBACK"] = resolved.physical_seq_len
    return resolved


class _CompatibilityModule(types.ModuleType):
    """Propagate legacy module-level assignments to canonical modules."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name.startswith("__"):
            return
        for target in _TARGET_MODULES:
            if hasattr(target, name):
                setattr(target, name, value)


sys.modules[__name__].__class__ = _CompatibilityModule
__all__ = sorted(
    {
        name
        for module in _TARGET_MODULES
        for name in getattr(module, "__all__", ())
    }
)
__all__.append("configure_runtime_sequence_length")
