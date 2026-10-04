#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python3 - <<'PY'
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

class StartupTests(unittest.TestCase):
    def run_start(self, interpreter):
        env = {**os.environ, 'ANALYZER_PYTHON': interpreter, 'BACKEND_PORT': 'invalid'}
        return subprocess.run(['bash', 'start.sh'], env=env, capture_output=True, text=True, timeout=5)

    def test_old_python_is_rejected_before_starting_services(self):
        with tempfile.TemporaryDirectory() as folder:
            interpreter = Path(folder) / 'python38'
            interpreter.write_text('#!/bin/sh\nexit 1\n')
            interpreter.chmod(0o755)
            result = self.run_start(str(interpreter))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Python 3.11', result.stderr)
        self.assertNotIn('Starting Python Sidecar', result.stdout)

    def test_missing_interpreter_has_actionable_error(self):
        result = self.run_start('/missing/archanalyzer-python')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ANALYZER_PYTHON', result.stderr)

    def test_supported_python_reaches_normal_port_validation(self):
        import sys
        result = self.run_start(sys.executable)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('BACKEND_PORT must be an integer', result.stderr)
        self.assertNotIn('Python 3.11', result.stderr)
        self.assertNotIn('Starting Python Sidecar', result.stdout)

unittest.main()
PY
