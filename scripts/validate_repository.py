"""A  fast validation script that checks the syntax of the submissions."""

from __future__ import annotations

import ast
import py_compile
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    col: int
    rule: str
    message: str

    def github_error(self) -> str:
        rel = self.path.relative_to(ROOT)
        return (
            f"::error file={rel},line={self.line},col={self.col},"
            f"title={self.rule}::{self.message}"
        )


def fail(message: str) -> None:
    print(f"::error::{message}")
    raise SystemExit(1)


def tracked_python_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node

    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value

    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))

    return None


def string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None

class AntiCheatVisitor(ast.NodeVisitor):
    """Reject syntactic access to competition-internal names prefixed with "_"."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[Finding] = []

    def add(self, node: ast.AST, rule: str, message: str) -> None:
        self.findings.append(
            Finding(
                path=self.path,
                line=getattr(node, "lineno", 1),
                col=getattr(node, "col_offset", 0) + 1,
                rule=rule,
                message=message,
            )
        )

    @staticmethod
    def is_hidden(name: str | None) -> bool:
        return bool(name) and name.startswith("_")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Examples:
        #   evaluator._ground_truth
        #   result._reference
        if self.is_hidden(node.attr):
            self.add(
                node,
                "SEC001",
                f"access to hidden/private attribute {node.attr!r} is not allowed",
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Examples:
        #   state["_ground_truth"]
        #   os.environ["_PRIVATE_REFERENCE"]
        key = string_literal(node.slice)
        if self.is_hidden(key):
            self.add(
                node,
                "SEC002",
                f"access to hidden/private key {key!r} is not allowed",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)

        # getattr(obj, "_hidden"), setattr(...), delattr(...)
        if name in {"getattr", "setattr", "delattr", "hasattr"} and len(node.args) >= 2:
            attr = string_literal(node.args[1])
            if self.is_hidden(attr):
                self.add(
                    node,
                    "SEC003",
                    f"dynamic access to hidden/private attribute {attr!r} is not allowed",
                )

        # vars(obj).get("_hidden"), dict.get("_hidden"), os.getenv("_HIDDEN"), etc.
        if name in {"os.getenv", "os.putenv", "getenv", "putenv"} and node.args:
            key = string_literal(node.args[0])
            if self.is_hidden(key):
                self.add(
                    node,
                    "SEC004",
                    f"access to hidden/private environment name {key!r} is not allowed",
                )

        # mapping.get("_hidden"), mapping.pop("_hidden"), mapping.setdefault("_hidden", ...)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get", "pop", "setdefault"}
            and node.args
        ):
            key = string_literal(node.args[0])
            if self.is_hidden(key):
                self.add(
                    node,
                    "SEC002",
                    f"access to hidden/private key {key!r} is not allowed",
                )

        self.generic_visit(node)


def validate_project_metadata() -> None:
    pyproject = ROOT / "pyproject.toml"
    lockfile = ROOT / "uv.lock"

    if not pyproject.is_file():
        fail("pyproject.toml is missing")
    if not lockfile.is_file():
        fail("uv.lock is missing; run `uv lock` and commit the lockfile")

    try:
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        fail(f"pyproject.toml could not be parsed: {exc}")

    project = data.get("project")
    if not isinstance(project, dict):
        fail("pyproject.toml has no [project] table")

    name = project.get("name", "<unnamed>")
    requires_python = project.get("requires-python", "<not specified>")
    print(f"Project: {name}")
    print(f"Python requirement: {requires_python}")


def validate_python_syntax() -> list[Path]:
    files = tracked_python_files()
    if not files:
        print("No tracked Python files found.")
        return []

    failures: list[str] = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc.msg}")

    if failures:
        for failure in failures:
            print(f"::error::{failure}")
        raise SystemExit(1)

    print(f"Syntax check passed for {len(files)} tracked Python file(s).")
    return files


def validate_static_anti_cheat(files: list[Path]) -> None:
    findings: list[Finding] = []
    algorithms_root = (ROOT / "clarify" / "algorithms").resolve()

    algorithm_files: list[Path] = []
    for path in files:
        try:
            path.resolve().relative_to(algorithms_root)
        except ValueError:
            continue
        algorithm_files.append(path)

    for path in algorithm_files:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            # Syntax errors are already handled above; this is defensive.
            fail(f"Could not statically inspect {path.relative_to(ROOT)}: {exc}")

        visitor = AntiCheatVisitor(path)
        visitor.visit(tree)
        findings.extend(visitor.findings)

    if findings:
        print("Static anti-cheat validation failed:")
        for finding in findings:
            print(finding.github_error())
        print(
            "These checks are syntactic only and do not prove that arbitrary "
            "Python is safe. Final evaluation must still be sandboxed."
        )
        raise SystemExit(1)

    print(
        "Static anti-cheat check passed for "
        f"{len(algorithm_files)} Python file(s) under clarify/algorithms."
    )


def main() -> int:
    validate_project_metadata()
    files = validate_python_syntax()
    validate_static_anti_cheat(files)
    print("Repository validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())