"""Processing worker: runs a recipe for ImSwitch on another computer, over plain HTTP.

    python worker.py --port 8090        (docker/processing-worker builds an image of it)

ImSwitch sends one volume and fetches the result. The worker never connects to the microscope, so
it only has to be reachable from the Pi. Jobs run one at a time, in arrival order; a result is kept
until it is deleted, at most an hour after the job ended.

    POST   /jobs?recipe=deconvolution&shape=Z,Y,X&dtype=<u2&spec_bytes=N
           body: N bytes of JSON spec, then the C-order pixels -> 201 {"id", "state": "queued", ..}
    GET    /jobs/<id>          -> {"id", "state", "progress", "error", "seconds"}
    GET    /jobs/<id>/result   -> float32 little endian, same shape, once the state is "done"
    DELETE /jobs/<id>          -> stops a queued or running job and forgets it
    GET    /                   -> {"worker": "imswitch-processing", "recipes": {name: version}}

``spec`` is what the recipe's ``run`` takes ({"optics": .., "parameters": ..} for deconvolution,
plus a measured PSF); it travels in the body because it can hold an array.
numpy and the recipe's own imports only, no ImSwitch imports. ``run_remote`` is the client.
"""

import argparse
import itertools
import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np

try:
    from . import deconvolution
except ImportError:  # a flat copy next to deconvolution.py (the container)
    import deconvolution

RECIPES = {deconvolution.NAME: deconvolution}
KEEP_S = 3600  # finished jobs are forgotten after an hour
MAX_BYTES = 16 << 30  # one volume
MAX_SPEC = 64 << 20  # the JSON spec, a measured PSF included


class Worker:
    """The jobs, and one thread that runs them in turn."""

    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()
        self.todo = queue.Queue()
        threading.Thread(target=self._loop, daemon=True, name="processing-worker").start()

    def submit(self, recipe: str, spec: dict, volume: np.ndarray) -> dict:
        job = {
            "id": uuid.uuid4().hex[:12],
            "recipe": recipe,
            "spec": spec,
            "volume": volume,
            "state": "queued",
            "progress": 0.0,
            "error": None,
            "seconds": None,
            "ended": None,
            "result": None,
            "stop": False,
        }
        with self.lock:
            now = time.monotonic()
            for key in [
                k for k, j in self.jobs.items() if j["ended"] and now - j["ended"] > KEEP_S
            ]:
                del self.jobs[key]
            self.jobs[job["id"]] = job
        self.todo.put(job)
        return self.status(job["id"])

    def status(self, key: str):
        job = self.jobs.get(key)
        return (
            None
            if job is None
            else {k: job[k] for k in ("id", "state", "progress", "error", "seconds")}
        )

    def drop(self, key: str) -> bool:
        with self.lock:
            job = self.jobs.pop(key, None)
        if job is not None:
            job["stop"] = True
        return job is not None

    def _loop(self):
        while True:
            job = self.todo.get()
            if job["stop"]:
                continue
            job["state"], started = "running", time.monotonic()

            def progress(fraction, job=job):
                job["progress"] = round(fraction, 4)
                return job["stop"]

            try:
                job["result"] = RECIPES[job["recipe"]].run(job.pop("volume"), job["spec"], progress)
                job["state"], job["progress"] = "done", 1.0
            except Exception as e:
                job["state"] = "cancelled" if job["stop"] else "failed"
                job["error"] = None if job["stop"] else f"{type(e).__name__}: {e}"
            job["seconds"], job["ended"] = round(time.monotonic() - started, 2), time.monotonic()


class Handler(BaseHTTPRequestHandler):
    worker: Worker = None  # set by handler()
    timeout = 120  # s per socket read or write: a stalled client frees its thread

    def _reply(self, code: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _parts(self):
        return urlparse(self.path).path.strip("/").split("/")

    def do_GET(self):
        parts = self._parts()
        if parts == [""]:
            return self._reply(
                200,
                {
                    "worker": "imswitch-processing",
                    "jobs": len(self.worker.jobs),
                    "recipes": {n: r.VERSION for n, r in RECIPES.items()},
                },
            )
        job = self.worker.jobs.get(parts[1]) if len(parts) > 1 and parts[0] == "jobs" else None
        if job is None or len(parts) > 3 or (len(parts) == 3 and parts[2] != "result"):
            return self._reply(404, {"error": "no such job"})
        if len(parts) == 2:
            return self._reply(200, self.worker.status(parts[1]))
        if job["state"] != "done":
            return self._reply(409, {"error": f"the job is {job['state']}"})
        result = job["result"]
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(result.size * 4))
        self.end_headers()
        for plane in result.reshape((-1,) + result.shape[-2:]):  # one plane at a time
            self.wfile.write(np.ascontiguousarray(plane, "<f4").tobytes())

    def _job_request(self):
        """recipe, shape, dtype, spec bytes and pixel bytes of a POST /jobs; ValueError if bad."""
        query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        recipe = query.get("recipe", deconvolution.NAME)
        shape = tuple(int(n) for n in query["shape"].split(","))
        dtype = np.dtype(query["dtype"])
        spec_bytes = int(query.get("spec_bytes", 0))
        size = int(np.prod(shape)) * dtype.itemsize
        if recipe not in RECIPES:
            raise ValueError(f"unknown recipe {recipe!r}; this worker has {sorted(RECIPES)}")
        if not (
            2 <= len(shape) <= 3
            and min(shape) >= 1
            and dtype.kind in "uif"
            and dtype.itemsize <= 8
            and 0 <= spec_bytes <= MAX_SPEC
        ):
            raise ValueError(
                "expected shape Z,Y,X or Y,X, an integer or float dtype, a spec of "
                f"{MAX_SPEC >> 20} MiB at most"
            )
        if int(self.headers.get("Content-Length", -1)) != spec_bytes + size:
            raise ValueError("Content-Length does not match spec_bytes, shape and dtype")
        return recipe, shape, dtype, spec_bytes, size

    def do_POST(self):
        if urlparse(self.path).path.strip("/") != "jobs":
            return self._reply(404, {"error": "not found"})
        read = 0
        try:
            recipe, shape, dtype, read, size = self._job_request()
            if size > MAX_BYTES:
                self._drain()
                return self._reply(
                    413, {"error": f"a volume may have at most {MAX_BYTES >> 30} GiB"}
                )
            spec = json.loads(self.rfile.read(read) or b"{}")
            if not isinstance(spec, dict):
                raise ValueError("the spec is not a JSON object")
        except (KeyError, ValueError, TypeError) as e:
            self._drain(read)
            return self._reply(400, {"error": str(e)})
        volume = np.empty(shape, dtype)
        view, got = memoryview(volume).cast("B"), 0
        while got < size:
            n = self.rfile.readinto(view[got:])
            if not n:
                return self._reply(400, {"error": "the body ended early"})
            got += n
        self._reply(201, self.worker.submit(recipe, spec, volume))

    def _drain(self, done: int = 0):
        """Read and drop the rest of the request body (``done`` bytes are read already), so the
        client gets the answer, not a broken pipe."""
        try:
            left = int(self.headers.get("Content-Length", 0)) - done
        except ValueError:
            return
        while 0 < left <= MAX_BYTES + MAX_SPEC:
            chunk = self.rfile.read(min(left, 1 << 20))
            if not chunk:
                break
            left -= len(chunk)

    def do_DELETE(self):
        parts = self._parts()
        found = len(parts) == 2 and parts[0] == "jobs" and self.worker.drop(parts[1])
        self._reply(200 if found else 404, {"deleted": bool(found)})

    def log_message(self, fmt, *args):  # the status polls would flood the log
        if self.command != "GET" or (len(args) > 1 and str(args[1]) != "200"):
            super().log_message(fmt, *args)


def handler(worker: Worker = None):
    """A request handler class bound to its own Worker."""
    return type("BoundHandler", (Handler,), {"worker": worker or Worker()})


# --- client -----------------------------------------------------------------------------------


def _call(request, timeout):
    """JSON answer of a worker request; RuntimeError with the worker's message on an HTTP error."""
    try:
        with urlopen(request, timeout=timeout) as r:
            return json.load(r)
    except HTTPError as e:
        raise RuntimeError(
            f"worker answered {e.code}: {e.read().decode(errors='replace')[:300]}"
        ) from e
    except URLError as e:
        url = getattr(request, "full_url", request).split("?")[0]
        raise RuntimeError(f"worker at {url} did not answer: {e.reason}") from e


def info(url: str, timeout: float = 3.0) -> dict:
    """What the worker at ``url`` offers; RuntimeError when it does not answer."""
    return _call(f"{url.rstrip('/')}/", timeout)


def run_remote(
    url, planes, shape, dtype, spec, recipe=deconvolution.NAME, progress=None, poll_s=1.0
):
    """Run ``recipe`` on one (z, y, x) volume at the worker ``url``; yields the result planes.

    ``planes`` yields the volume's (y, x) planes. They are sent as they come and the result comes
    back plane by plane (float32), so the caller never holds a whole volume. ``progress(fraction)``
    works as in deconvolve(): returning True cancels the job and raises Cancelled. RuntimeError
    when the worker reports a failure.
    """
    url, (nz, ny, nx) = url.rstrip("/"), shape
    dt = np.dtype(dtype).newbyteorder("<")
    head = json.dumps(spec).encode()
    query = urlencode(
        {"recipe": recipe, "shape": f"{nz},{ny},{nx}", "dtype": dt.str, "spec_bytes": len(head)}
    )
    body = itertools.chain([head], (np.ascontiguousarray(plane, dt).tobytes() for plane in planes))
    key = _call(
        Request(
            f"{url}/jobs?{query}",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(head) + nz * ny * nx * dt.itemsize),
            },
        ),
        timeout=600,
    )["id"]
    try:
        while True:
            job = _call(f"{url}/jobs/{key}", timeout=30)
            if job["state"] == "done":
                break
            if job["state"] in ("failed", "cancelled"):
                raise RuntimeError(f"worker: {job['error'] or job['state']}")
            if progress is not None and progress(job["progress"]):
                raise deconvolution.Cancelled()
            time.sleep(poll_s)
        with urlopen(f"{url}/jobs/{key}/result", timeout=600) as r:
            for _ in range(nz):
                plane = bytearray(ny * nx * 4)
                view, got = memoryview(plane), 0
                while got < len(plane):
                    n = r.readinto(view[got:])
                    if not n:
                        raise RuntimeError("worker: the result ended early")
                    got += n
                yield np.frombuffer(plane, "<f4").reshape(ny, nx)
    finally:
        try:
            urlopen(Request(f"{url}/jobs/{key}", method="DELETE"), timeout=10).close()
        except OSError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), handler())
    print(
        f"processing worker on http://{args.host}:{args.port}, recipes: {', '.join(RECIPES)}",
        flush=True,
    )
    server.serve_forever()
