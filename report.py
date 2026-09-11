"""Build or serve the fit report.

  report.py serve                 regenerate the data and serve it on localhost
  report.py build                 one standalone .html with the data inlined
  report.py data                  just write report_data.json

`serve` is the one to use while working. It serves a directory containing only the page
and its data, so nothing else in this directory -- the parquet, the codings, the pickles
-- is reachable over the socket, and it regenerates the data whenever the fit is newer
than the last build, so refitting and reloading the browser is the whole loop.

`build` inlines the data into a single file for sending to someone or publishing. That is
the only difference between the two: the page is the same template either way and decides
at run time which mode it is in.
"""

import http.server
import json
import shutil
import socketserver
import threading
import time
import webbrowser
from pathlib import Path

import typer

import data as D
import report_data

app = typer.Typer(help=__doc__, add_completion=False)

TEMPLATE = Path("report_template.html")
DATA = Path("report_data.json")
TAG = '<script id="D" type="application/json"></script>'


def stale(pkl: Path, data: Path) -> bool:
    return not data.exists() or data.stat().st_mtime < pkl.stat().st_mtime


def ensure_data(pkl: Path, codings: Path, out: Path, force: bool = False,
                parquet: Path = D.PARQUET) -> bool:
    """Rebuild `out` if the fit is newer than it. Returns whether it rebuilt."""
    if not force and not stale(pkl, out):
        return False
    report_data.build(pkl=str(pkl), codings=str(codings), out=str(out),
                      parquet=str(parquet))
    return True


@app.command()
def data(
    pkl: Path = typer.Option(Path("fit.pkl")),
    codings: Path = typer.Option(D.CODINGS),
    parquet: Path = typer.Option(D.PARQUET),
    out: Path = typer.Option(DATA),
) -> None:
    """Write report_data.json from a saved fit."""
    report_data.build(pkl=str(pkl), codings=str(codings), out=str(out),
                      parquet=str(parquet))


@app.command()
def build(
    pkl: Path = typer.Option(Path("fit.pkl")),
    codings: Path = typer.Option(D.CODINGS),
    parquet: Path = typer.Option(D.PARQUET),
    out: Path = typer.Option(Path("fit_report.html")),
    refresh: bool = typer.Option(True, "--refresh/--no-refresh",
                                 help="rebuild the data first"),
) -> None:
    """One standalone HTML file with the data inlined -- for sending or publishing."""
    if refresh:
        report_data.build(pkl=str(pkl), codings=str(codings), out=str(DATA),
                          parquet=str(parquet))
    html = TEMPLATE.read_text()
    if TAG not in html:
        raise SystemExit(f"{TEMPLATE} has no empty data tag to fill; expected {TAG}")
    blob = DATA.read_text()
    html = html.replace(TAG, f'<script id="D" type="application/json">{blob}</script>')
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, data inlined)")


@app.command()
def serve(
    pkl: Path = typer.Option(Path("fit.pkl")),
    codings: Path = typer.Option(D.CODINGS),
    parquet: Path = typer.Option(D.PARQUET),
    port: int = typer.Option(8000),
    host: str = typer.Option("127.0.0.1", help="loopback by default; pass 0.0.0.0 to "
                                               "expose the report on your network"),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
    root: Path = typer.Option(Path(".report"), help="the directory actually served"),
) -> None:
    """Serve the report on localhost, rebuilding the data when the fit changes."""
    root.mkdir(exist_ok=True)
    if not TEMPLATE.exists():
        raise SystemExit(f"{TEMPLATE} not found; run this from the project directory")
    if not pkl.exists():
        raise SystemExit(f"{pkl} not found; run `fit.py run` first")

    served_data = root / "report_data.json"
    ensure_data(pkl, codings, served_data, force=True, parquet=parquet)
    shutil.copyfile(TEMPLATE, root / "index.html")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def do_GET(self):
            # A refit while the server is up should need nothing but a browser reload.
            if self.path in ("/", "/index.html"):
                shutil.copyfile(TEMPLATE, root / "index.html")
                try:
                    if ensure_data(pkl, codings, served_data, parquet=parquet):
                        print(f"  {pkl} changed -- data rebuilt")
                except Exception as e:            # a bad refit must not kill the server
                    print(f"  could not rebuild data: {type(e).__name__}: {e}")
            return super().do_GET()

        def guess_type(self, path):
            # SimpleHTTPRequestHandler sends "text/html" with no charset, and a browser
            # that falls back to Latin-1 turns every multi-byte character into mojibake.
            # The page is ASCII-only as a second line of defence, but say it anyway.
            t = super().guess_type(path)
            base = t.split(";")[0].strip()
            if base in ("text/html", "application/json", "text/plain", "text/css",
                        "application/javascript", "text/javascript"):
                return base + "; charset=utf-8"
            return t

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, fmt, *args):
            if "200" not in (args[1] if len(args) > 1 else ""):
                super().log_message(fmt, *args)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    url = f"http://{'localhost' if host == '127.0.0.1' else host}:{port}/"
    try:
        srv = Server((host, port), Handler)
    except OSError as e:
        raise SystemExit(f"cannot bind {host}:{port} ({e}). Try --port {port + 1}.")
    print(f"serving {root}/ at {url}")
    print(f"  fit      {pkl}")
    print(f"  codings  {codings}")
    print("  reload the page after a refit; the data rebuilds itself. ctrl-c to stop.")
    if open_browser:
        threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url)),
                         daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()


if __name__ == "__main__":
    app()
