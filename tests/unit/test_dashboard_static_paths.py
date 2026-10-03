import re
from pathlib import Path

STATIC_DIR = Path(__file__).parents[2] / "src/prime_rl/dashboard/static"
ABSOLUTE_REF = re.compile(r'(?:href|src)="/(?!/)')


def test_index_html_has_no_absolute_refs():
    html = (STATIC_DIR / "index.html").read_text()
    assert not ABSOLUTE_REF.search(html)


def test_app_js_has_no_absolute_api_paths():
    js = (STATIC_DIR / "app.js").read_text()
    assert 'fetch(path)' not in js
    assert 'EventSource("/api/' not in js
