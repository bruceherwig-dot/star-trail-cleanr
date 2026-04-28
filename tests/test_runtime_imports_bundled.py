"""Every top-level external import in worker code must be in the frozen bundle.

Regression test for the v1.81-beta crash: scikit-image was imported at module
top in modules/detect_trails.py but not bundled in build_helper.py because the
`--collect-all skimage` line was stashed away while shipping a "clean" v1.81.
Every new install crashed on Batch 1 with ModuleNotFoundError.

This test statically scans every worker file's top-level imports, classifies
each as stdlib / project-internal / external, and asserts every external
package is either in build_helper.py's --collect-all list OR in the explicit
KNOWN_OK_WITHOUT_COLLECT_ALL allowlist below.

Lazy imports (inside a function or class body) are intentionally excluded —
those are the right pattern for optional dependencies and don't need bundling
at all if the gating code never runs in production.
"""
import ast
import importlib.util
import re
import sys
import sysconfig
from pathlib import Path

REPO = Path(__file__).parent.parent

# Files whose top-level imports must resolve in the frozen bundle. The GUI,
# the worker entrypoint, and every modules/*.py the worker can import.
WORKER_FILES = [
    REPO / "astro_clean_v5.py",
    REPO / "star_trail_cleanr.py",
    REPO / "mask_painter.py",
    REPO / "modules" / "detect_trails.py",
    REPO / "modules" / "repair.py",
    REPO / "modules" / "hot_pixels.py",
    REPO / "modules" / "slope_match.py",
    REPO / "modules" / "clean_sky.py",
    REPO / "modules" / "align.py",
    REPO / "modules" / "model_update.py",
    REPO / "modules" / "user_folder.py",
]

# External packages PyInstaller auto-detection bundles cleanly without
# --collect-all. Add to this list ONLY after verifying the frozen build
# launches and uses the package without ModuleNotFoundError.
KNOWN_OK_WITHOUT_COLLECT_ALL = {
    # Pillow (import name PIL) has shipped in every release since v1.0-beta
    # without needing --collect-all. PyInstaller's import scanner handles its
    # subpackages and bundled fonts cleanly.
    "PIL",
    # Sentry SDK uses standard import patterns; auto-detection works.
    "sentry_sdk",
    # PyTorch is bundled by --collect-all transitively via ultralytics, plus
    # it's installed from the CPU-only wheel index in CI which lays it out
    # in a way PyInstaller scans cleanly.
    "torch",
    "torchvision",
    # NVIDIA ML Python bindings — Windows-only optional dep, lazy-loaded.
    "nvidia_ml_py",
    "pynvml",
}

# Build-helper-only / dev-mode-only modules that aren't part of the worker
# runtime and shouldn't be required in the frozen bundle. Imports of these
# are excluded from the check.
DEV_ONLY_MODULES = {"_sentry_config"}


def _is_builtin_or_frozen(spec):
    if spec is None or spec.origin is None:
        return spec is not None
    return spec.origin in ("built-in", "frozen")


def _classify(name: str):
    """Return one of 'stdlib', 'project', 'external', 'unresolved'."""
    if name in DEV_ONLY_MODULES:
        return "project"
    # Project-internal: a file with that name exists in the repo.
    if (REPO / f"{name}.py").exists() or (REPO / name).is_dir():
        return "project"
    # 3.10+ has sys.stdlib_module_names. Older Pythons fall back to find_spec.
    stdlib_names = getattr(sys, "stdlib_module_names", None)
    if stdlib_names is not None:
        if name in stdlib_names:
            return "stdlib"
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return "unresolved"
    if spec is None:
        return "unresolved"
    if _is_builtin_or_frozen(spec):
        return "stdlib"
    origin = spec.origin or ""
    if origin.startswith(sysconfig.get_paths()["stdlib"]):
        return "stdlib"
    return "external"


def _top_level_imports(path: Path):
    """Yield (package_name, lineno) for every module-level import statement.
    Imports inside def/class bodies are skipped — those are lazy by design."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                continue  # relative import, project-internal
            if node.module:
                yield node.module.split(".")[0], node.lineno
        elif isinstance(node, ast.If):
            # Top-level `if sys.platform == 'win32': import X` style guards.
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        yield alias.name.split(".")[0], sub.lineno
                elif isinstance(sub, ast.ImportFrom) and sub.level == 0 and sub.module:
                    yield sub.module.split(".")[0], sub.lineno


def _collect_all_packages_from_build_helper():
    """Parse build_helper.py to extract the list of packages passed to
    --collect-all. The flag and the package name are separate strings in the
    cmd list, so we look for `'--collect-all', 'X'` pairs."""
    text = (REPO / "build_helper.py").read_text(encoding="utf-8")
    pkgs = set()
    pattern = re.compile(r"'--collect-all'\s*,\s*'([^']+)'")
    pkgs.update(pattern.findall(text))
    return pkgs


def test_every_external_top_level_import_is_bundled():
    collect_all = _collect_all_packages_from_build_helper()
    bundled = collect_all | KNOWN_OK_WITHOUT_COLLECT_ALL

    missing = []  # (file, lineno, package)
    for f in WORKER_FILES:
        if not f.exists():
            continue
        for pkg, lineno in _top_level_imports(f):
            cls = _classify(pkg)
            if cls in ("stdlib", "project"):
                continue
            if cls == "unresolved":
                # Package not installed in this environment. CI installs all
                # runtime deps before running tests, so unresolved here means
                # an import we never use — flag it as missing.
                missing.append((f.name, lineno, pkg, "unresolved"))
                continue
            if pkg not in bundled:
                missing.append((f.name, lineno, pkg, "external"))

    if missing:
        lines = [
            "Top-level imports not covered by the frozen bundle:",
        ]
        for fname, lineno, pkg, cls in missing:
            lines.append(f"  {fname}:{lineno}  {pkg!r}  ({cls})")
        lines += [
            "",
            "Each missing package needs ONE of:",
            "  1. Add `--collect-all <package>` to build_helper.py.",
            "  2. Move the import inside a function or class body so it is "
            "lazy (only do this if the call site is gated off in production).",
            "  3. Add the package to KNOWN_OK_WITHOUT_COLLECT_ALL in this "
            "test file, but ONLY after building and confirming the frozen "
            "app launches and exercises the package without crashing.",
            "",
            "This regression bit v1.81-beta when scikit-image was top-level "
            "imported but not bundled. Every new install crashed on Batch 1.",
        ]
        raise AssertionError("\n".join(lines))


def test_collect_all_list_in_build_helper_is_parseable():
    """Sanity check on the parsing helper itself: build_helper.py contains
    --collect-all entries, and the parser finds at least the ones we
    historically rely on."""
    pkgs = _collect_all_packages_from_build_helper()
    for required in ("cv2", "numpy", "PySide6", "sahi", "ultralytics"):
        assert required in pkgs, \
            f"build_helper.py is missing --collect-all {required}"
