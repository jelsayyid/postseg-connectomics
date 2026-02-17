"""Tests for _build_dataclass nested-dataclass and tuple-field branches.

IMPORTANT: This file intentionally omits `from __future__ import annotations`
so that field types are stored as actual type objects (not strings). This is
required to reach lines 128 and 130 in utils/config.py, which check
`dataclasses.is_dataclass(f.type)` and `f.type == tuple` respectively.
"""

from dataclasses import dataclass, field

from connectomics_pipeline.utils.config import _build_dataclass


@dataclass
class InnerCfg:
    x: int = 0
    y: str = "default"


@dataclass
class TupleCfg:
    tup: tuple = (1, 2, 3)
    val: int = 0


@dataclass
class OuterCfg:
    inner: InnerCfg = field(default_factory=InnerCfg)
    count: int = 0


class TestBuildDataclassNestedDataclass:
    def test_nested_dataclass_recursed(self):
        """Line 128: _build_dataclass recurses into nested dataclass field."""
        result = _build_dataclass(OuterCfg, {"inner": {"x": 7, "y": "hi"}, "count": 3})
        assert isinstance(result.inner, InnerCfg)
        assert result.inner.x == 7
        assert result.inner.y == "hi"
        assert result.count == 3

    def test_nested_dataclass_none_value(self):
        """Nested field with None data returns default-constructed dataclass."""
        result = _build_dataclass(OuterCfg, {"inner": None})
        assert isinstance(result.inner, InnerCfg)
        assert result.inner.x == 0


class TestBuildDataclassTupleField:
    def test_list_converted_to_tuple(self):
        """Line 130: a list value for a tuple-typed field is converted to tuple."""
        result = _build_dataclass(TupleCfg, {"tup": [10, 20, 30]})
        assert result.tup == (10, 20, 30)
        assert isinstance(result.tup, tuple)

    def test_tuple_value_preserved(self):
        """A tuple value for a tuple-typed field passes through unchanged."""
        result = _build_dataclass(TupleCfg, {"tup": (7, 8, 9)})
        assert result.tup == (7, 8, 9)
