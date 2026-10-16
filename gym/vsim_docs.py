"""Serve the prebuilt vlearn (VSim) Sphinx docs locally: `uv run vsim-docs`."""

import argparse
import functools
import http.server
import os
import sys
import webbrowser

from gym import GYM_ROOT_DIR

DOCS_DIR = os.path.join(GYM_ROOT_DIR, "thirdparty", "vlearn", "docs")
HTML_DIR = os.path.join(DOCS_DIR, "vlearn-docs", "html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(HTML_DIR, "index.html")):
        sys.exit(
            f"vlearn docs not found at {HTML_DIR}\n"
            f"Unzip them first: unzip {DOCS_DIR}/vlearn-docs.zip -d {DOCS_DIR}"
        )

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=HTML_DIR
    )
    # Bind to localhost only; the docs are licensed material.
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/"
        print(f"Serving vlearn docs at {url} (Ctrl+C to stop)", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
