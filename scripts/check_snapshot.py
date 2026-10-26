"""Check the offline source inventory without importing integration modules."""

from __future__ import annotations

import ast
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ADDRESS = re.compile(
    r"[a-z][a-z0-9+.-]*:[/]{2}\S+|www\.|"
    r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|"
    r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|cn|io|ai)\b",
    re.I,
)
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
LINK = re.compile(r"!?\[[^\]\n]*\]\([^\n]*\)|<(?:a|img)\b|\b(?:href|src)\s*=", re.I)


def source_files(root: Path) -> list[str]:
    manifest = root / 'SOURCE_MANIFEST.json'
    if manifest.is_file():
        payload = json.loads(manifest.read_text())
        if payload.get('schema_version') != 'anonymous-source-archive-v1':
            raise ValueError('source_manifest_schema_invalid')
        names = sorted([*payload['files'], 'SOURCE_MANIFEST.json'])
        if (root / '.git').exists():
            tracked = set(filter(None, subprocess.check_output(
                ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                cwd=root, text=True,
            ).split('\0')))
            actual = {name for name in tracked if (root / name).is_file()}
            if actual != set(names):
                raise ValueError('source_inventory_mismatch')
        return names
    if not (root / '.git').exists():
        raise ValueError('source_inventory_missing')
    return sorted(set(filter(None, subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=root, text=True,
    ).split('\0'))))


def check_snapshot(root: Path = ROOT) -> dict:
    root = root.resolve()
    inventory = root / 'SOURCE_MANIFEST.json'
    expected = json.loads(inventory.read_text())['files'] if inventory.is_file() else None
    files = 0
    modules = 0
    failures = []
    for name in source_files(root):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or '.git' in relative.parts:
            failures.append((name, 'unsafe_path'))
            continue
        path = root / relative
        if not path.exists():
            if expected is not None:
                failures.append((name, 'missing_file'))
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            failures.append((name, 'symlink_or_outside_root'))
            continue
        raw = path.read_bytes()
        files += 1
        if (expected is not None and name != 'SOURCE_MANIFEST.json'
                and hashlib.sha256(raw).hexdigest() != expected[name]):
            failures.append((name, 'hash_mismatch'))
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            failures.append((name, 'unexpected_binary'))
            continue
        if ADDRESS.search(text) or ADDRESS.search(name):
            failures.append((name, 'address'))
        for token in IPV4.findall(text):
            try:
                address = ipaddress.ip_address(token)
            except ValueError:
                continue
            if not (address.is_loopback or address.is_unspecified):
                failures.append((name, 'network_address'))
        if path.suffix == '.md' and LINK.search(text):
            failures.append((name, 'documentation_link'))
        if path.suffix == '.py':
            ast.parse(text, filename=name)
            modules += 1
    if failures:
        raise ValueError(json.dumps(failures, ensure_ascii=True))
    return {'files': files, 'source_modules': modules, 'address_hits': 0,
            'documentation_link_hits': 0, 'syntax_valid': True,
            'execution_scope': 'review_tools_only'}


if __name__ == '__main__':
    print(json.dumps(check_snapshot(), indent=2))
