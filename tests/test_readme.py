"""Keep README.md consistent with the code, the recorded metrics and the test suite."""

import ast
import json
import re
from pathlib import Path

import pytest

from movie_genre.config import METRICS_PATH, PROJECT_ROOT, SVC_C

README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
SOURCE_DIR = PROJECT_ROOT / "src" / "movie_genre"
TESTS_DIR = Path(__file__).resolve().parent


def percent(value):
    return f"{value * 100:.2f}%"


def cli_options():
    """Every --option declared with add_argument in the package, as (module, option)."""
    options = []
    for path in sorted(SOURCE_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                        options.append((path.stem, arg.value))
    return options


def test_readme_documents_every_cli_option():
    # The lookahead stops --output from being satisfied by --output-dir.
    missing = [f"{module}: {option}" for module, option in cli_options() if not re.search(re.escape(option) + r"(?![\w-])", README)]
    assert not missing, f"Options not mentioned in README.md: {missing}"


@pytest.mark.skipif(not METRICS_PATH.exists(), reason="metrics.json not available")
def test_readme_reports_recorded_metrics():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    shipped = metrics["shipped_model"]
    expected = [
        percent(metrics["majority_class_baseline"]),
        *(percent(shipped["test"][key]) for key in ("accuracy", "weighted_f1", "macro_f1", "top3_accuracy")),
        *(percent(shipped["test_no_title"][key]) for key in ("accuracy", "weighted_f1", "macro_f1")),
    ]
    assert [value for value in expected if value not in README] == []

    # Model selection table: val accuracy, val weighted F1, test accuracy, test weighted F1, test macro F1.
    rows = {(row["model"], row["split"]): row for row in metrics["metrics"] if row["fit"] == "train split (80%)"}
    for model in {model for model, _ in rows}:
        line = next((line for line in README.splitlines() if line.startswith(f"| {model} |")), None)
        assert line is not None, f"No selection table row for {model}"
        val, test = rows[(model, "validation")], rows[(model, "held-out test")]
        wanted = [val["accuracy"], val["weighted_f1"], test["accuracy"], test["weighted_f1"], test["macro_f1"]]
        assert re.findall(r"\d+\.\d+%", line) == [percent(value) for value in wanted], model


def test_readme_reports_tuned_c():
    assert f"`C={SVC_C:g}`" in README


def test_readme_test_count_matches_suite():
    # Counts test functions; revisit this if parametrized tests are added.
    count = sum(
        isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        for path in TESTS_DIR.glob("test_*.py")
        for node in ast.parse(path.read_text(encoding="utf-8")).body
    )
    stated = {int(number) for number in re.findall(r"(\d+) tests\b", README)}
    assert stated == {count}, f"README states {sorted(stated)} tests, suite has {count}"


def test_readme_relative_links_exist():
    targets = re.findall(r"\]\(([^)\s]+)\)", README) + re.findall(r'src="([^"]+)"', README)
    local = {target for target in targets if not re.match(r"[a-z]+:|#", target)}
    assert local, "expected at least one relative link"
    assert [target for target in sorted(local) if not (PROJECT_ROOT / target).exists()] == []
