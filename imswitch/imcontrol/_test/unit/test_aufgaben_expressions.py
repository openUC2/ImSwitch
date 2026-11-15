"""The Aufgaben expression language: what it computes and what it refuses."""
import pytest

from imswitch.imcontrol.model.aufgaben.expressions import ExpressionError, evaluate, references

NS = {"param": {"limit": 3, "name": "A1"}, "var": {"n": 2, "xs": [1, 2, 3]},
      "loop": {"t": {"index": 4, "count": 10}}, "result": {"af": {"z_um": 12.5}, "bad": None},
      "device": {"temperature_c": 36.8}}


@pytest.mark.parametrize("text,expected", [
    ("n + 1", 3), ("var.n * param.limit", 6), ("n < param.limit and not False", True),
    ("loop.t.index % 2 == 0", True), ("result.af.z_um - 0.5", 12.0),
    ("result.bad is None", True), ("device.temperature_c >= 37 - 0.5", True),
    ("xs[1]", 2), ("max(xs) + len(xs)", 6), ("mean(xs)", 2.0),
    ("'A' in param.name", True), ("1 if n > 1 else 0", 1), ("[n, n]", [2, 2]),
    ("1 < n < 3", True),
])
def test_evaluates(text, expected):
    assert evaluate(text, NS) == expected


@pytest.mark.parametrize("text", [
    "__import__('os')", "n.__class__", "().__class__.__bases__", "lambda: 1",
    "[x for x in xs]", "open('f')", "eval('1')", "getattr(n, 'real')", "f(x=1)",
    "n := 3", "xs.pop()", "_secret", "",
])
def test_refuses(text):
    with pytest.raises(ExpressionError):
        evaluate(text, NS)


def test_huge_powers_are_refused():
    with pytest.raises(ExpressionError):
        evaluate("2 ** 1000000", NS)


def test_unknown_names_fail_clearly():
    with pytest.raises(ExpressionError, match="'missing' is not defined"):
        evaluate("missing + 1", NS)


def test_references_by_namespace():
    refs = references("var.a + b > param.c and result.af.z_um and loop.t.index")
    assert refs["var"] == {"a", "b"}
    assert refs["param"] == {"c"}
    assert refs["result"] == {"af"}
    assert refs["loop"] == {"t"}
