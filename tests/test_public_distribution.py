"""Public distribution boundaries and documentation stay usable after syncing."""

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_public_tree_excludes_private_outputs_and_duplicate_release_layouts():
    tracked = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=ROOT, text=True,
    ).split('\0')
    forbidden = {'release', '.hl', '.planning', '.claude', 'backups', 'experiments',
                 'batch_results', 'results', 'reports', 'outputs'}
    for name in filter(None, tracked):
        path = Path(name)
        if not (ROOT / path).is_file():
            continue
        assert path.parts[0] not in forbidden, name
        assert not name.startswith('scenarios/operate_v'), name
        assert not (path.parent == Path('docs') and path.name.startswith('LATEST_RESULTS'))


def test_public_markdown_links_resolve():
    paths = [*ROOT.glob('*.md'), *(ROOT / 'docs').rglob('*.md'),
             ROOT / 'scripts/README.md', ROOT / 'benchmark/README.md',
             ROOT / 'evaluation/policies/README.md', ROOT / 'data/README.md']
    for path in paths:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            target = target.split('#', 1)[0]
            if target and '://' not in target and not target.startswith('mailto:'):
                assert (path.parent / target).exists(), (path, target)


def test_container_context_excludes_local_configuration_and_workspaces():
    ignored = set((ROOT / '.dockerignore').read_text().splitlines())
    assert {'.git', '.hl', '.planning', '.claude', '.codex', '.env', '.env.*',
            '.envrc', 'api_config_local.py'} <= ignored
