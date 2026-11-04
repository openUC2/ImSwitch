"""A small, safe expression language for job parameters and conditions.

Python expression syntax, evaluated by walking a whitelisted AST: literals,
arithmetic, comparisons, boolean logic, ``a if c else b``, lists and tuples,
attribute and index access on plain data, and a few functions. No ``eval``, no
imports, no calls on objects, no names starting with an underscore.

Namespace at run time (all plain data):

- ``param.<name>``   job parameters
- ``var.<name>``     job variables; a bare ``name`` also resolves to a variable
- ``loop.<id>``      ``index``, ``count`` and ``item`` of an enclosing loop
- ``result.<id>``    what a node last returned (e.g. ``result.af``); a node skipped by
                     ``every`` or ``enabled`` keeps its previous result
- ``device.<...>``   readings the host exposes (position, temperature, inputs)
- ``job``            ``name``, ``run_id``, ``elapsed_s``
"""
from __future__ import annotations

import ast
import math
import operator
import time
from typing import Any, Callable, Dict, Mapping, Set

NAMESPACES = ("param", "var", "loop", "result", "device", "job")

FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "abs": abs, "min": min, "max": max, "len": len, "round": round,
    "int": int, "float": float, "str": str, "bool": bool, "sum": sum,
    "mean": lambda xs: sum(xs) / len(xs) if len(xs) else math.nan,
    "now": time.time,
}

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}
_CMP = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_, ast.IsNot: operator.is_not,
}
_ALLOWED = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.IfExp, ast.Constant, ast.Name, ast.Load, ast.Attribute, ast.Subscript, ast.List,
    ast.Tuple, ast.Call, *_BINOPS, *_UNARY, *_CMP,
)
MAX_POWER = 64
MAX_LENGTH = 500


class ExpressionError(ValueError):
    pass


def _check_node(node: ast.AST, text: str) -> None:
    if not isinstance(node, _ALLOWED):
        raise ExpressionError(f"{type(node).__name__} is not allowed in {text!r}")
    if isinstance(node, ast.Name) and node.id.startswith("_"):
        raise ExpressionError(f"name {node.id!r} is not allowed")
    if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
        raise ExpressionError(f"attribute {node.attr!r} is not allowed")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise ExpressionError(f"only {sorted(FUNCTIONS)} can be called")
        if node.keywords:
            raise ExpressionError("keyword arguments are not allowed")


def parse(text: str) -> ast.Expression:
    """Parse and check an expression; raises ExpressionError if not allowed."""
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("empty expression")
    if len(text) > MAX_LENGTH:
        raise ExpressionError(f"expression longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"syntax error in {text!r}: {e.msg}") from None
    for node in ast.walk(tree):
        _check_node(node, text)
    return tree


def references(text: str) -> Dict[str, Set[str]]:
    """Names an expression reads, by namespace: {'param': {...}, 'var': {...}, ...}.

    Bare names that are not namespaces or functions count as variables.
    """
    refs: Dict[str, Set[str]] = {ns: set() for ns in NAMESPACES}
    tree = parse(text)
    attr_bases = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in NAMESPACES):
            refs[node.value.id].add(node.attr)
            attr_bases.add(id(node.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and id(node) not in attr_bases:
            if node.id in NAMESPACES or node.id in FUNCTIONS:
                continue
            refs["var"].add(node.id)
    return refs


def evaluate(text: str, namespace: Mapping[str, Any]) -> Any:
    """Evaluate ``text`` against ``namespace`` (keys from NAMESPACES)."""
    return _eval(parse(text).body, namespace, text)


def _lookup(container: Any, key: Any, text: str) -> Any:
    if isinstance(container, Mapping):
        if key not in container:
            raise ExpressionError(f"{key!r} is not defined (in {text!r})")
        return container[key]
    if isinstance(key, str):
        if key.startswith("_") or not hasattr(container, key):
            raise ExpressionError(f"{key!r} is not defined (in {text!r})")
        return getattr(container, key)
    raise ExpressionError(f"cannot look up {key!r} (in {text!r})")


def _name(node, ns, text):
    if node.id in NAMESPACES:
        return ns.get(node.id, {})
    return _lookup(ns.get("var", {}), node.id, text)


def _subscript(node, ns, text):
    container = _eval(node.value, ns, text)
    key = _eval(node.slice, ns, text)
    try:
        return container[key]
    except (KeyError, IndexError, TypeError) as e:
        raise ExpressionError(f"{key!r}: {e} (in {text!r})") from None


def _sequence(node, ns, text):
    values = [_eval(e, ns, text) for e in node.elts]
    return values if isinstance(node, ast.List) else tuple(values)


def _boolop(node, ns, text):
    want = not isinstance(node.op, ast.And)   # And stops at the first falsy value
    result = not want
    for v in node.values:
        result = _eval(v, ns, text)
        if bool(result) == want:
            return result
    return result


def _binop(node, ns, text):
    left, right = _eval(node.left, ns, text), _eval(node.right, ns, text)
    if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > MAX_POWER:
        raise ExpressionError(f"exponent larger than {MAX_POWER} (in {text!r})")
    try:
        return _BINOPS[type(node.op)](left, right)
    except (ArithmeticError, TypeError) as e:
        raise ExpressionError(f"{e} (in {text!r})") from None


def _compare(node, ns, text):
    left = _eval(node.left, ns, text)
    for op, comp in zip(node.ops, node.comparators):
        right = _eval(comp, ns, text)
        try:
            if not _CMP[type(op)](left, right):
                return False
        except TypeError as e:
            raise ExpressionError(f"{e} (in {text!r})") from None
        left = right
    return True


def _call(node, ns, text):
    args = [_eval(a, ns, text) for a in node.args]
    try:
        return FUNCTIONS[node.func.id](*args)
    except (TypeError, ValueError, ArithmeticError) as e:
        raise ExpressionError(f"{node.func.id}(): {e} (in {text!r})") from None


_EVAL = {
    ast.Constant: lambda node, ns, text: node.value,
    ast.Name: _name,
    ast.Attribute: lambda node, ns, text: _lookup(_eval(node.value, ns, text), node.attr, text),
    ast.Subscript: _subscript,
    ast.List: _sequence,
    ast.Tuple: _sequence,
    ast.BoolOp: _boolop,
    ast.UnaryOp: lambda node, ns, text: _UNARY[type(node.op)](_eval(node.operand, ns, text)),
    ast.BinOp: _binop,
    ast.Compare: _compare,
    ast.IfExp: lambda node, ns, text: _eval(
        node.body if _eval(node.test, ns, text) else node.orelse, ns, text),
    ast.Call: _call,
}


def _eval(node: ast.AST, ns: Mapping[str, Any], text: str) -> Any:
    handler = _EVAL.get(type(node))
    if handler is None:
        raise ExpressionError(f"{type(node).__name__} is not allowed in {text!r}")
    return handler(node, ns, text)
