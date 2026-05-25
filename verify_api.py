import ast
import argparse
import importlib
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
API_FILE = ROOT / "api.py"
RUNNER_FILE = ROOT / "run_api.py"

REQUIRED_ROUTES = {
    ("get", "/health"),
    ("post", "/research"),
    ("get", "/research/{task_id}"),
    ("get", "/research/{task_id}/report"),
    ("get", "/research/{task_id}/events"),
}


def compile_files() -> None:
    for path in (API_FILE, RUNNER_FILE):
        py_compile.compile(str(path), doraise=True)


def collect_routes(tree: ast.AST) -> set[tuple[str, str]]:
    routes: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not isinstance(func, ast.Attribute):
                continue
            if not isinstance(func.value, ast.Name) or func.value.id != "api":
                continue
            if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
                continue
            if not isinstance(decorator.args[0].value, str):
                continue
            routes.add((func.attr.lower(), decorator.args[0].value))
    return routes


def verify_routes() -> None:
    tree = ast.parse(API_FILE.read_text(encoding="utf-8"), filename=str(API_FILE))
    routes = collect_routes(tree)
    missing = REQUIRED_ROUTES - routes
    if missing:
        missing_text = ", ".join(f"{method.upper()} {path}" for method, path in sorted(missing))
        raise AssertionError(f"Missing API routes: {missing_text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the FastAPI wrapper.")
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Also import api.py. This requires project runtime dependencies.",
    )
    args = parser.parse_args()

    compile_files()
    verify_routes()
    print("FastAPI wrapper structural verification passed.")

    if args.runtime:
        try:
            importlib.import_module("api")
        except ModuleNotFoundError as exc:
            missing = exc.name or "unknown"
            print(
                "Runtime import verification failed: "
                f"missing dependency '{missing}'. "
                "Install project dependencies with `pip install -r requirements.txt`.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
        print("FastAPI wrapper runtime import verification passed.")


if __name__ == "__main__":
    main()
