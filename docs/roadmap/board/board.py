"""Serve the WP board on localhost; PUT /wps.json rewrites the file, one WP per line.

    python3 docs/roadmap/board/board.py [port]           # default 8771
    python3 docs/roadmap/board/board.py export OUT.html  # one read-only page, e.g. for GitHub Pages
"""

import http.server
import json
import os
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "wps.json"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def do_PUT(self):
        if self.path != "/wps.json":
            return self.send_error(404)
        try:
            wps = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ids = [w["id"] for w in wps]
            if not all(isinstance(i, str) and i for i in ids) or len(ids) != len(set(ids)):
                raise ValueError("every WP needs a unique, non-empty id")
        except (ValueError, KeyError, TypeError) as e:
            return self.send_error(400, str(e))
        tmp = DATA.with_suffix(".tmp")
        lines = ",\n".join(json.dumps(w, ensure_ascii=False) for w in wps)
        tmp.write_text(f"[\n{lines}\n]\n", encoding="utf-8")
        os.replace(tmp, DATA)
        self.send_response(204)
        self.end_headers()


def export(out):
    """Write index.html with wps.json and people.json inlined, read-only."""
    page = HERE / "index.html"
    if out.resolve() == page:
        sys.exit("refusing to overwrite the board's own index.html")
    files = (DATA, HERE / "people.json")
    wps, roles = (f.read_text(encoding="utf-8").replace("</", "<\\/") for f in files)
    snap = f"{{date: '{date.today()}', wps: {wps}, roles: {roles}}}"
    snap = f"<script>window.SNAPSHOT = {snap};</script>\n</head>"
    html = page.read_text(encoding="utf-8")
    assert html.count("</head>") == 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("</head>", snap), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    if sys.argv[1:2] == ["export"]:
        if len(sys.argv) != 3:
            sys.exit("usage: board.py export OUT.html")
        export(Path(sys.argv[2]))
        sys.exit()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8771
    print(f"http://localhost:{port}/")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
