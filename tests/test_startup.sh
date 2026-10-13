#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python3 - <<'PY'
import os
import subprocess
import tempfile
import unittest
import shutil
from pathlib import Path

class StartupTests(unittest.TestCase):
    def run_fixture(self, certificates=(), fallback=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copyfile('start.sh', root / 'start.sh')
            (root / 'backend').mkdir()
            commands = root / 'commands'
            commands.mkdir()
            stubs = {
                'python': '#!/bin/sh\nif [ "$1" = "-c" ]; then exit 0; fi\nexec sleep 30\n',
                'curl': '#!/bin/sh\nexit 0\n',
                'lsof': '#!/bin/sh\nexit 1\n',
                'npm': '#!/bin/sh\nprintf "%s\\n" "$@" > "$STARTUP_TEST_ARGS"\n',
            }
            for name, source in stubs.items():
                command = commands / name
                command.write_text(source)
                command.chmod(0o755)
            if not fallback:
                next_command = root / 'node_modules/.bin/next'
                next_command.parent.mkdir(parents=True)
                next_command.write_text(stubs['npm'])
                next_command.chmod(0o755)
            if certificates:
                (root / '.certificates').mkdir()
                for name in certificates:
                    (root / '.certificates' / name).write_text('test certificate')
            args_file = root / 'next-args'
            env = {**os.environ, 'PATH': str(commands) + os.pathsep + os.environ['PATH'],
                   'ANALYZER_PYTHON': str(commands / 'python'), 'STARTUP_TEST_ARGS': str(args_file),
                   'BACKEND_PORT': '8080', 'FRONTEND_PORT': '3001', 'FRONTEND_HOST': '0.0.0.0'}
            result = subprocess.run(['bash', 'start.sh'], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=5)
            args = args_file.read_text().splitlines() if args_file.exists() else []
            return result, args

    def test_certificate_pair_enables_https_in_both_launch_paths(self):
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                result, args = self.run_fixture(('server.pem', 'server-key.pem'), fallback)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--experimental-https', args)
                self.assertIn('--experimental-https-cert', args)
                self.assertIn('--experimental-https-key', args)
                self.assertTrue(args[args.index('--experimental-https-cert') + 1].endswith('/.certificates/server.pem'))
                self.assertTrue(args[args.index('--experimental-https-key') + 1].endswith('/.certificates/server-key.pem'))
                self.assertIn('https://0.0.0.0:3001', result.stdout)

    def test_no_certificates_preserves_http(self):
        result, args = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('--experimental-https', args)
        self.assertIn('http://0.0.0.0:3001', result.stdout)

    def test_incomplete_certificate_pair_stops_before_starting_services(self):
        for name in ('server.pem', 'server-key.pem'):
            with self.subTest(name=name):
                result, args = self.run_fixture((name,))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(args, [])
                self.assertNotIn('Starting Python Sidecar', result.stdout)
                self.assertIn('server.pem', result.stderr)

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
