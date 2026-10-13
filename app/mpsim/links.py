"""Linked inputs: one varied quantity sets several entries of the form.

In a study the same physical quantity often appears in several places of the
form - an alumina blend of 10 um and 3 um particles is two phases with one
interfacial resistance, one contact resistance and one conductivity. Varied
or calibrated as separate factors they double the design for nothing and,
worse, let the two drift apart. A link ties further entries to a factor:

    phases.1.r_int      = x                 the same value
    phases.1.props.k    = 0.8 * x           a fixed ratio
    contact_rc.0-1      = sqrt(x1 * x2)     a function of several factors

Every DOE parameter, calibration unknown and so every optimisation factor
has a symbol: x1, x2, ... in the order they are listed, and `x` for the one
the link belongs to. A link's expression may use them all, numbers, + - * /
** and parentheses, and the functions below. Expressions are parsed, not
executed (no Python eval): only these names and operations exist. They are
evaluated element-wise on NumPy arrays as well, so the effective-medium
preview of a calibration evaluates its whole parameter grid at once.
"""
from __future__ import annotations

import ast
import math
import operator

import numpy as np

FUNCTIONS = {
    "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10,
    "abs": np.abs, "min": np.minimum, "max": np.maximum, "pow": np.power,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "tanh": np.tanh,
}
CONSTANTS = {"pi": math.pi, "e": math.e}
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_LEN = 200


class LinkError(ValueError):
    pass


def symbols(n):
    """The symbols of n factors: x1 ... xn."""
    return [f"x{i + 1}" for i in range(n)]


def parse(expr, names):
    """Check an expression and return its tree. `names`: the symbols it may
    use (x, x1, ...). Raises LinkError with a message for the user."""
    src = str(expr if expr is not None else "").strip() or "x"
    if len(src) > MAX_LEN:
        raise LinkError(f"The expression is longer than {MAX_LEN} characters")
    try:
        tree = ast.parse(src.replace("^", "**"), mode="eval")
    except SyntaxError:
        raise LinkError(f"'{src}' is not a valid expression") from None
    allowed = set(names) | set(CONSTANTS)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load)) or type(node) in _BIN or type(node) in _UN:
            continue
        if isinstance(node, (ast.BinOp, ast.UnaryOp)):
            if isinstance(node, ast.BinOp) and type(node.op) not in _BIN:
                raise LinkError(f"'{src}': only + - * / and ** (or ^) are allowed")
            if isinstance(node, ast.UnaryOp) and type(node.op) not in _UN:
                raise LinkError(f"'{src}': only + and - signs are allowed")
            continue
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise LinkError(f"'{src}': only numbers are allowed as constants")
            continue
        if isinstance(node, ast.Name):
            if node.id not in allowed and node.id not in FUNCTIONS:
                raise LinkError(f"'{src}': unknown name '{node.id}' (use {', '.join(sorted(names))}, "
                                f"pi, e and {', '.join(sorted(FUNCTIONS))})")
            continue
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                raise LinkError(f"'{src}': only the functions {', '.join(sorted(FUNCTIONS))} can be called")
            continue
        raise LinkError(f"'{src}': '{type(node).__name__}' is not allowed in an expression")
    return tree


def _eval(node, env):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        return CONSTANTS[node.id]
    if isinstance(node, ast.BinOp):
        return _BIN[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.UnaryOp):
        return _UN[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.Call):
        return FUNCTIONS[node.func.id](*[_eval(a, env) for a in node.args])
    raise LinkError(f"'{type(node).__name__}' is not allowed")


def evaluate(expr, env):
    """Value of an expression for the symbols in env (floats or arrays)."""
    tree = parse(expr, list(env))
    try:
        with np.errstate(all="ignore"):
            v = _eval(tree, env)
    except (ZeroDivisionError, OverflowError, ValueError, TypeError) as e:
        raise LinkError(f"'{expr}' cannot be evaluated for {_env_txt(env)} ({e})") from None
    a = np.asarray(v, float)
    if not np.all(np.isfinite(a)):
        raise LinkError(f"'{expr}' is not a finite number for {_env_txt(env)}")
    return float(a) if a.ndim == 0 else a


def _env_txt(env):
    out = []
    for k, v in env.items():
        a = np.asarray(v, float)
        out.append(f"{k} = {float(a):.4g}" if a.ndim == 0 else f"{k} = {float(a.min()):.4g}...{float(a.max()):.4g}")
    return ", ".join(out)


def factor_links(factors):
    """[(index of the factor, target path, expression)] of a list of factors
    (DOE parameters or calibration unknowns), each with an optional list
    "links": [{"path": ..., "expr": ...}]."""
    out = []
    for j, f in enumerate(factors):
        for ln in f.get("links") or []:
            path = (ln.get("path") or "").strip()
            if path:
                out.append((j, path, str(ln.get("expr") or "x").strip() or "x"))
    return out


def check(factors, primary_paths=None, allowed=None):
    """Validate the links of these factors: every expression parses with the
    symbols there are, no entry is set twice (by two links, or by a link and
    a factor), and - with `allowed` (path -> reason or None) - every target
    is one the caller can set. Returns a list of messages (empty: fine)."""
    errs = []
    names = ["x"] + symbols(len(factors))
    prim = list(primary_paths if primary_paths is not None else [f.get("path") for f in factors])
    seen = {p: f"factor x{i + 1}" for i, p in enumerate(prim) if p}
    for j, path, expr in factor_links(factors):
        who = f"x{j + 1}"
        try:
            parse(expr, names)
        except LinkError as e:
            errs.append(f"Link of {who} to {path}: {e}")
        if path in seen:
            errs.append(f"{path} is set by {seen[path]} and again by a link of {who}")
        seen[path] = f"a link of {who}"
        if allowed is not None:
            why = allowed(path)
            if why:
                errs.append(f"Link of {who} to {path}: {why}")
    return errs


def linked_values(factors, values):
    """{target path: value} of every link for the factor values given in
    order (floats, or arrays of equal length for a whole grid)."""
    env = {s: v for s, v in zip(symbols(len(factors)), values)}
    out = {}
    for j, path, expr in factor_links(factors):
        e = dict(env, x=values[j])
        try:
            out[path] = evaluate(expr, e)
        except LinkError as err:
            raise LinkError(f"Link of x{j + 1} to {path}: {err}") from None
    return out
