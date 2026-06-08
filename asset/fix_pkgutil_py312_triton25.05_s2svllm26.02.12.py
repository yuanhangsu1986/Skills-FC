"""
Python 3.12 compatibility fix for pkg_resources (old setuptools).

Two incompatibilities confirmed inside the triton25.05_s2svllm26.02.12.sqsh container:

  1. pkgutil.ImpImporter  — removed in Python 3.12.  Old pkg_resources registers it
     as an import finder at module level:
       register_finder(pkgutil.ImpImporter, find_on_path)
       register_namespace_handler(pkgutil.ImpImporter, file_ns_handler)

  2. importer.find_module() — removed in Python 3.12.  Old pkg_resources falls back
     to it in _handle_ns when find_spec() returns None:
       except AttributeError:
           loader = importer.find_module(packageName)

Both the LOCAL  (/usr/local/lib/python3.12/dist-packages/pkg_resources/__init__.py)
and SYSTEM (/usr/lib/python3/dist-packages/pkg_resources/__init__.py) copies have
these lines.  Both are writable inside the container overlay.  We patch both.
"""
import pathlib
import re
import sys

# ── Replacements (exact string matches on the confirmed bad lines) ────────────

# Fix 1: ImpImporter register_finder calls
# Before: register_finder(pkgutil.ImpImporter, find_on_path)
# After:  if hasattr(pkgutil, 'ImpImporter'): register_finder(pkgutil.ImpImporter, find_on_path)
IMPIMP_FINDER = (
    "register_finder(pkgutil.ImpImporter,",
    "if hasattr(pkgutil, 'ImpImporter'): register_finder(pkgutil.ImpImporter,",
)

# Fix 2: ImpImporter register_namespace_handler call
IMPIMP_NS = (
    "register_namespace_handler(pkgutil.ImpImporter,",
    "if hasattr(pkgutil, 'ImpImporter'): register_namespace_handler(pkgutil.ImpImporter,",
)

# Fix 3: _handle_ns try/except that uses find_spec().loader then falls back to
# find_module().  The try block crashes if find_spec() returns None; the except
# block calls find_module() which was removed in 3.12.
#
# Local pkg_resources uses this exact indentation and wording:
_HANDLE_NS_OLD_LOCAL = """\
    try:
        loader = importer.find_spec(packageName).loader
    except AttributeError:
        # capture warnings due to #1111
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = importer.find_module(packageName)
"""

# System (Debian) pkg_resources uses slightly different wording:
_HANDLE_NS_OLD_SYSTEM = """\
    try:
        loader = importer.find_spec(packageName).loader
    except AttributeError:
        # capture warnings due to #1111
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = importer.find_module(packageName)
"""

# Both are identical text so one replacement covers both.
_HANDLE_NS_NEW = """\
    try:
        _spec = importer.find_spec(packageName)
        loader = _spec.loader if _spec is not None else None
    except AttributeError:
        # find_module() removed in Python 3.12; use find_spec with None guard
        _spec = getattr(importer, 'find_spec', lambda n, p=None: None)(packageName)
        loader = _spec.loader if _spec is not None else None
"""

# Catch-all for any remaining bare importer.find_module( calls (safety net)
_FIND_MODULE_RE = re.compile(r'\bimporter\.find_module\s*\(')
_FIND_MODULE_REPLACEMENT = (
    "getattr(importer, 'find_module', lambda n: None)("
)


def patch_file(path: pathlib.Path) -> bool:
    try:
        original = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[fix_pkgutil_py312] Cannot read {path}: {e}")
        return False

    content = original

    # Apply fixes in order
    content = content.replace(IMPIMP_FINDER[0], IMPIMP_FINDER[1])
    content = content.replace(IMPIMP_NS[0], IMPIMP_NS[1])
    content = content.replace(_HANDLE_NS_OLD_LOCAL, _HANDLE_NS_NEW)
    content = _FIND_MODULE_RE.sub(_FIND_MODULE_REPLACEMENT, content)

    if content == original:
        print(f"[fix_pkgutil_py312] Already clean (no changes): {path}")
        return False

    try:
        path.write_text(content, encoding="utf-8")
    except Exception as e:
        print(f"[fix_pkgutil_py312] Cannot write {path}: {e}")
        return False

    # Remove stale .pyc so Python recompiles from patched source
    cache_dir = path.parent / "__pycache__"
    for pyc in cache_dir.glob(f"{path.stem}*.pyc"):
        try:
            pyc.unlink()
        except Exception:
            pass

    print(f"[fix_pkgutil_py312] Patched: {path}")
    return True


# Patch every pkg_resources/__init__.py reachable from sys.path
patched_any = False
seen = set()
for sp in sys.path:
    candidate = pathlib.Path(sp) / "pkg_resources" / "__init__.py"
    if candidate in seen or not candidate.exists():
        continue
    seen.add(candidate)
    if patch_file(candidate):
        patched_any = True

if not patched_any:
    print("[fix_pkgutil_py312] Nothing to patch — all pkg_resources copies already clean.")
