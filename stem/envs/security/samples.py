"""Intentionally vulnerable sample code for the security-audit benchmark.

This code is NOT meant to run. Each snippet contains planted defects of known categories;
``GROUND_TRUTH`` records them so the environment can score an audit objectively. This is a
defensive/benchmark artifact, not an attack tool.
"""

from __future__ import annotations

CATEGORIES = [
    "sql_injection",
    "command_injection",
    "path_traversal",
    "hardcoded_secret",
    "weak_crypto",
    "insecure_random",
]

SAMPLES: dict[str, str] = {
    "auth.py": '''\
import sqlite3

API_TOKEN = "sk_live_9f8d7a6b5c4e3f2a1b0c"  # planted: hardcoded secret

def login(db, username, password):
    cur = db.cursor()
    # planted: sql injection (string-formatted query)
    cur.execute("SELECT * FROM users WHERE name = '%s' AND pw = '%s'" % (username, password))
    return cur.fetchone()
''',
    "files.py": '''\
import os

def fetch(filename):
    # planted: path traversal (no sanitisation of user path)
    path = "/var/data/" + filename
    with open(path) as f:
        return f.read()

def backup(target):
    # planted: command injection (unsanitised input to shell)
    os.system("tar czf /tmp/backup.tgz " + target)
''',
    "tokens.py": '''\
import hashlib
import random

def password_hash(pw):
    # planted: weak crypto (md5 for passwords)
    return hashlib.md5(pw.encode()).hexdigest()

def session_token():
    # planted: insecure random for security token
    return "".join(random.choice("0123456789abcdef") for _ in range(16))
''',
    "report.py": '''\
import subprocess

def render(db, user_id, output_name):
    cur = db.cursor()
    # planted: sql injection
    cur.execute("SELECT * FROM reports WHERE owner = " + str(user_id))
    rows = cur.fetchall()
    # planted: command injection
    subprocess.call("wkhtmltopdf - " + output_name, shell=True)
    return rows
''',
}

GROUND_TRUTH: dict[str, set[str]] = {
    "auth.py": {"hardcoded_secret", "sql_injection"},
    "files.py": {"path_traversal", "command_injection"},
    "tokens.py": {"weak_crypto", "insecure_random"},
    "report.py": {"sql_injection", "command_injection"},
}

# train/test split over the files; categories overlap so a learned audit skill transfers.
TRAIN_FILES = ["auth.py", "files.py"]
TEST_FILES = ["tokens.py", "report.py"]
