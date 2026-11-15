#!/usr/bin/env python3
"""Benchmark: a scan canvas (OME-Zarr, with its pyramid) into mikro on an Arkitekt hub.

    python scripts/arkitekt_canvas_bench.py make 10 10 --out /home/pi/Datasets/arkitekt-bench
    python scripts/arkitekt_canvas_bench.py upload <canvas.ome.zarr> --hub 111

``make`` writes a synthetic RGB scan of FRAME ID10 camera tiles (3072 x 2048) tile
by tile with ImSwitch's NgffImage, then builds its pyramid, as the ExperimentController
does. ``upload`` logs in at the coordination server (device code; cached after the first
approval, a fresh code whenever one expires) and uploads the canvas with
imswitch.imcontrol.model.io.mikro_upload, as ImSwitch does. Each prints one JSON line:
timings and peak memory. See docs/arkitekt-hub-pi.md for the figures.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

TILE_H, TILE_W = 2048, 3072


def peak_rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def synthetic_tile(rng, c: int, ix: int, iy: int) -> np.ndarray:
    """A 12-bit tile: background, noise, a few hundred 'cells', a bright border."""
    tile = (300 + 40 * c + 60 * ((ix + iy) % 3)) * np.ones((TILE_H, TILE_W), np.float32)
    tile += rng.normal(0, 25, size=(TILE_H, TILE_W)).astype(np.float32)
    for _ in range(300):
        cy, cx, r = rng.integers(20, TILE_H - 20), rng.integers(20, TILE_W - 20), rng.integers(4, 18)
        y = np.arange(cy - r, cy + r + 1)[:, None]
        x = np.arange(cx - r, cx + r + 1)[None, :]
        tile[cy - r:cy + r + 1, cx - r:cx + r + 1][(y - cy) ** 2 + (x - cx) ** 2 <= r * r] += \
            rng.uniform(500, 3000)
    tile[:16, :] = tile[-16:, :] = tile[:, :16] = tile[:, -16:] = 4000  # shows the placement
    return np.clip(tile, 0, 4095).astype(np.uint16)


def make(args) -> dict:
    from imswitch.imcontrol.model.aufgaben.storage import ImageSpec, NgffImage
    nx, ny = args.nx, args.ny
    path = args.out / f"canvas_{nx}x{ny}_{args.channels}c_{time.strftime('%H%M%S')}.ome.zarr"
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ["LED_rgb_r", "LED_rgb_g", "LED_rgb_b"] if args.channels == 3 else \
        [f"C{i}" for i in range(args.channels)]
    spec = ImageSpec(shape=(1, args.channels, 1, ny * TILE_H, nx * TILE_W), dtype="uint16",
                     pixel_size_um=args.pixel_um, origin_um=(*args.origin_um, 0.0),
                     channel_names=names, name=path.name, chunks_yx=(TILE_H, TILE_W),
                     bit_depth=12)
    image = NgffImage.create(path, spec)
    rng = np.random.default_rng(0)
    started = time.time()
    for iy in range(ny):
        for ix in range(nx):
            for c in range(args.channels):
                image.write(synthetic_tile(rng, c, ix, iy), 0, c, 0, iy * TILE_H, ix * TILE_W)
    written = time.time()
    image.finalize()
    return {"canvas": str(path), "tiles": nx * ny, "write_s": round(written - started, 1),
            "pyramid_s": round(time.time() - written, 1), "levels": image.levels}


def upload(args) -> dict:
    import dask
    from arkitekt import App, connect
    from mikro import Mikro, mikro_service
    from imswitch.imcontrol.model.io.mikro_upload import read_ngff, stage_space, upload_ngff

    canvas = read_ngff(args.canvas)
    result = {"shape": list(canvas.levels[0].shape), "raw_gb": round(canvas.nbytes / 1e9, 2),
              "levels": len(canvas.levels), "workers": args.workers}

    async def show_code(challenge):
        link = challenge.verification_uri_complete or challenge.endpoint
        if args.hub:
            link += ("&" if "?" in link else "?") + f"hub={args.hub}"
        print(f"APPROVE {link} (code {challenge.user_code}, {challenge.expires_in} s)", flush=True)

    app = App("imswitch-ingest-test", "0.1.0", description="ImSwitch canvas upload benchmark",
              services=[mikro_service])
    for attempt in range(12):  # a fresh code whenever one expires unapproved
        try:
            started = time.time()
            with connect(app, url=args.url, headless=True, device_code_hook=show_code) as rt:
                result["login_s"] = round(time.time() - started, 1)
                mikro = rt.require(Mikro)
                started = time.time()
                with dask.config.set(scheduler="threads", num_workers=args.workers):
                    dataset = upload_ngff(mikro, canvas, space=stage_space(mikro))
                result.update(upload_s=round(time.time() - started, 1), dataset=str(dataset.id))
            return result
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ != "DeviceCodeExpiredError" or attempt == 11:
                raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make", help="write a synthetic scan canvas and its pyramid")
    m.add_argument("nx", type=int)
    m.add_argument("ny", type=int)
    m.add_argument("--channels", type=int, default=3)
    m.add_argument("--pixel-um", type=float, default=0.5)
    m.add_argument("--origin-um", nargs=2, type=float, default=(1000.0, 2000.0))
    m.add_argument("--out", type=Path, default=Path("/home/pi/Datasets/arkitekt-bench"))
    u = sub.add_parser("upload", help="upload a canvas to mikro")
    u.add_argument("canvas", type=Path)
    u.add_argument("--url", default="https://go.arkitekt.live")
    u.add_argument("--hub", default="", help="hub id to preselect on the approval page")
    u.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    result = make(args) if args.command == "make" else upload(args)
    print("RESULT " + json.dumps({**result, "peak_rss_mb": peak_rss_mb()}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
