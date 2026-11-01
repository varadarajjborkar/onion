
import json
import os
import subprocess
import sys

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from review import build


# -----------------------------------------
# YOUR CODE: build the review page from the output of phases 1 to 5 and open it
# usage: python pipeline/run_review.py [--no-open] [--build-only]
# Your marks are saved to processed/review/feedback.json as you click. Press Ctrl-C to stop.
# -----------------------------------------

PORT = 8765


class Handler(SimpleHTTPRequestHandler):

    def do_POST(self):

        if self.path != "/feedback":
            self.send_error(404)
            return

        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))

        try:
            json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400)
            return

        # Write beside the target, then swap in one step, so a cut never leaves a torn file
        path = Path(self.directory) / "feedback.json"
        tmp = path.with_name("feedback.json.tmp")

        tmp.write_bytes(body)
        os.replace(tmp, path)

        self.send_response(204)
        self.end_headers()

    def end_headers(self):

        # The page must always read the newest marks
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


page = build()

print(f"Review page: {page}  ({page.stat().st_size / 1e6:.1f} MB)")

if "--build-only" in sys.argv:
    raise SystemExit

server = ThreadingHTTPServer(("127.0.0.1", PORT), partial(Handler, directory=str(page.parent)))

url = f"http://127.0.0.1:{PORT}/"

print(f"Open {url}   (marks are saved to {page.parent / 'feedback.json'} as you click; Ctrl-C to stop)")

if "--no-open" not in sys.argv:
    subprocess.run(["open", url])

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nStopped.")
