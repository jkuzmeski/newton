# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Find exact and near function clones without importing the shoe projects.

This is a review tool, not an automatic refactorer. Matching source structure
cannot establish equal physical assumptions, globals, dtype rules or provenance.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECTS = ("digital_instron_v2", "digital_shoe", "impedance_instron")


class _Locals(ast.NodeTransformer):
    """Normalize arguments and assigned local names, retaining operations/constants."""

    def __init__(self, node):
        names = [a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
        names.extend(a.arg for a in (node.args.vararg, node.args.kwarg) if a is not None)
        for item in ast.walk(node):
            if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)) and item.id not in names:
                names.append(item.id)
        self.names = {name: f"local_{i}" for i, name in enumerate(names)}

    def visit_Name(self, node):
        result = copy.copy(node)
        result.id = self.names.get(node.id, node.id)
        return result


def _body(node):
    statements = list(node.body)
    if statements and isinstance(statements[0], ast.Expr):
        value = statements[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            statements.pop(0)
    return ast.Module(body=statements, type_ignores=[])


def _functions(tree, prefix=""):
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{prefix}.{node.name}" if prefix else node.name
            yield name, node
            yield from _functions(node, name)
        elif isinstance(node, ast.ClassDef):
            name = f"{prefix}.{node.name}" if prefix else node.name
            yield from _functions(node, name)
        else:
            yield from _functions(node, prefix)


def _tokens(node):
    result = []
    for child in ast.walk(node):
        value = type(child).__name__
        if isinstance(child, ast.Name):
            value += ":" + child.id
        elif isinstance(child, ast.Attribute):
            value += ":" + child.attr
        elif isinstance(child, ast.Constant):
            value += ":" + repr(child.value)
        result.append(value)
    return result


def scan(source_root: Path, *, min_nodes: int = 20, near_threshold: float = 0.88, near_limit: int = 40):
    """Return review candidates, retaining constants and distinguishing thin adapters."""
    rows = []
    for project in PROJECTS:
        for path in sorted((source_root / "projects" / project).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for name, node in _functions(ast.parse(text, filename=str(path))):
                body = _body(node)
                nodes = sum(1 for _ in ast.walk(body))
                if nodes < min_nodes:
                    continue
                normal = _Locals(node).visit(copy.deepcopy(body))
                calls = sum(isinstance(item, ast.Call) for item in ast.walk(body))
                wrapper = len(body.body) == 1 and calls == 1 and isinstance(body.body[0], (ast.Return, ast.Expr))
                rows.append(
                    {
                        "path": path.relative_to(source_root).as_posix(),
                        "function": name,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                        "nodes": nodes,
                        "thin_adapter": wrapper,
                        "_exact": ast.dump(body, include_attributes=False),
                        "_normal": ast.dump(normal, include_attributes=False),
                        "_tokens": _tokens(normal),
                    }
                )
    groups = defaultdict(list)
    for row in rows:
        groups[row["_normal"]].append(row)

    def public(row):
        return {key: value for key, value in row.items() if not key.startswith("_")}

    duplicates = []
    for group in groups.values():
        if len({row["path"] for row in group}) < 2:
            continue
        duplicates.append(
            {
                "kind": "exact" if len({row["_exact"] for row in group}) == 1 else "renamed_locals",
                "all_thin_adapters": all(row["thin_adapter"] for row in group),
                "functions": [public(row) for row in group],
            }
        )
    near = []
    if near_limit:
        for i, a in enumerate(rows):
            if a["thin_adapter"]:
                continue
            for b in rows[i + 1 :]:
                if b["thin_adapter"] or a["path"] == b["path"] or a["_normal"] == b["_normal"]:
                    continue
                if min(a["nodes"], b["nodes"]) / max(a["nodes"], b["nodes"]) < near_threshold:
                    continue
                matcher = difflib.SequenceMatcher(None, a["_tokens"], b["_tokens"], autojunk=False)
                if matcher.quick_ratio() < near_threshold:
                    continue
                score = matcher.ratio()
                if score >= near_threshold:
                    near.append({"similarity": score, "a": public(a), "b": public(b)})
        near.sort(key=lambda item: (-item["similarity"], -min(item["a"]["nodes"], item["b"]["nodes"])))
        near = near[:near_limit]
    return {
        "function_count": len(rows),
        "min_nodes": min_nodes,
        "exact_or_renamed_groups": duplicates,
        "near_threshold": near_threshold,
        "near_candidates": near,
        "qualification": "Candidates require semantic review; different boundaries, globals or dataflow can be intentional.",
    }


def unreviewed_groups(result, review):
    """Return new algorithm clone groups not covered by the explicit review manifest."""
    allowed = [set(group["functions"]) for group in review.get("groups", [])]
    unknown = []
    for group in result["exact_or_renamed_groups"]:
        if group["all_thin_adapters"]:
            continue
        identifiers = {f"{row['path']}:{row['function']}" for row in group["functions"]}
        if not any(identifiers <= known for known in allowed):
            unknown.append(group)
    return unknown


def main():
    """Write a reproducible duplicate review; never execute project functions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-nodes", type=int, default=20)
    parser.add_argument("--near-threshold", type=float, default=0.88)
    parser.add_argument("--near-limit", type=int, default=40)
    parser.add_argument(
        "--check", action="store_true", help="Fail on exact/renamed algorithm groups without a reviewed exception."
    )
    parser.add_argument(
        "--review-manifest", type=Path, help="Default: projects/shoe_duplicate_review.json under source root."
    )
    args = parser.parse_args()
    if args.min_nodes < 1 or not 0 < args.near_threshold <= 1 or args.near_limit < 0:
        parser.error("require min-nodes >= 1, 0 < near-threshold <= 1 and near-limit >= 0")
    result = scan(
        args.source_root.resolve(),
        min_nodes=args.min_nodes,
        near_threshold=args.near_threshold,
        near_limit=args.near_limit,
    )
    unknown = []
    if args.check:
        manifest = args.review_manifest or args.source_root / "projects/shoe_duplicate_review.json"
        review = json.loads(manifest.read_text(encoding="utf-8"))
        unknown = unreviewed_groups(result, review)
        result["unreviewed_exact_groups"] = unknown
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    print(
        f"Reviewed {result['function_count']} functions; "
        f"{len(result['exact_or_renamed_groups'])} exact/renamed groups, "
        f"{len(result['near_candidates'])} near-match candidates.",
        file=sys.stderr,
    )
    if unknown:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
