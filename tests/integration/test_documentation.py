"""Check local Markdown links in both current guides and the historical archive."""
from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]


def documents():
    yield from ROOT.glob('*.md')
    for directory in ('apps', 'src', 'configs', 'data', 'results', 'tests', 'requirements', 'docs', 'archive'):
        yield from (ROOT/directory).rglob('*.md')


class DocumentationTests(unittest.TestCase):
    def test_local_markdown_links_resolve(self):
        failures = []
        for path in documents():
            # Code examples and URL links are not local documentation targets.
            text = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
            targets = re.findall(r'\[[^\]]*\]\(<?([^\s)>]+)>?(?:\s+"[^"]*")?\)', text)
            targets += re.findall(r'^\s*\[[^\]]+\]:\s*<?([^\s>]+)', text, flags=re.M)
            for url in targets:
                parsed = urlsplit(url)
                if parsed.scheme or parsed.netloc or url.startswith('#'):
                    continue
                target = path.parent/unquote(parsed.path)
                if not target.exists():
                    failures.append(f'{path.relative_to(ROOT)} -> {url}')
        self.assertEqual(failures, [], '\n'.join(failures))
