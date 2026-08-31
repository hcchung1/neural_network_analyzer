#!/usr/bin/env python3
"""Quick verification script for ArchAnalyzer backend."""

import ast
import sys
import unittest
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

def check_syntax(filepath: Path) -> bool:
    """Check Python syntax of a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        print(f"  ✓ {filepath.name}")
        return True
    except SyntaxError as e:
        print(f"  ✗ {filepath.name}: Syntax error at line {e.lineno}: {e.msg}")
        return False

def verify_imports(filepath: Path) -> bool:
    """Verify that imports are well-formed."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    # Check for common issues
    for node in imports:
        if isinstance(node, ast.ImportFrom) and node.module:
            if ' ' in node.module or any(ord(c) > 127 for c in node.module):
                print(f"  ⚠ {filepath.name}: Suspicious import '{node.module}'")
    return True

def main():
    print("Scanning ArchAnalyzer backend...\n")

    py_files = list(backend_dir.rglob('*.py'))
    if not py_files:
        print("No Python files found!", file=sys.stderr)
        sys.exit(1)

    all_ok = True
    for f in py_files:
        if 'pycache' in str(f):
            continue
        all_ok &= check_syntax(f)
        verify_imports(f)

    # Run compatibility unit tests
    print("\nRunning unit tests...")
    loader = unittest.TestLoader()
    suite = loader.discover(str(backend_dir), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=1)
    test_result = runner.run(suite)
    all_ok &= test_result.wasSuccessful()

    print(f"\n{'='*40}")
    if all_ok:
        print("✓ All backend files syntax OK and tests passed!")
    else:
        print("✗ Some files have syntax errors or tests failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
