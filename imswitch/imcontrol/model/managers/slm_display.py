"""Output sinks for the SLM / DMD display: where a pattern ends up as pixels.

Three backends, chosen by ``slm.display`` in the setup JSON:

* ``virtual``      - nothing is shown; the frame is kept for previews (tests, the
                     virtual microscope).
* ``framebuffer``  - Linux ``/dev/fbN`` written directly (stdlib only). The
                     production path on the Raspberry Pi: the HDMI output *is*
                     the DMD input, no X/Wayland needed.
* ``window``       - a borderless fullscreen window on monitor ``displayIndex``
                     via pygame-ce (SDL2), run in a *subprocess* so the server
                     never hosts a GUI toolkit (macOS insists on the main thread
                     for windows) and a toolkit crash cannot take ImSwitch down.
                     ``pip install pygame-ce`` (extra ``imswitch[slm]``).

Every backend has ``size`` (width, height), ``show(frame)`` which returns once
the frame is on the output, and ``close()``. ``frame`` is uint8 HxW (grey) or
HxWx3 (RGB) at the display size.

This file is also the window subprocess: ``python slm_display.py --window ...``.
It therefore imports only numpy and the stdlib at module level.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import threading
from typing import List, Optional, Tuple

import numpy as np

FB_IOGET_VSCREENINFO = 0x4600
FB_IOPUT_VSCREENINFO = 0x4601
FBIO_WAITFORVSYNC = 0x4004460B
FB_ACTIVATE_FORCE = 128
_VSCREENINFO_LEN = 160  # struct fb_var_screeninfo


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 from HxW grey or HxWx3/4 colour."""
    a = np.asarray(frame)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return np.repeat(a[:, :, None], 3, axis=2)
    return np.ascontiguousarray(a[:, :, :3])


class VirtualDisplay:
    """No output; remembers the last frame."""

    def __init__(self, width: int = 1920, height: int = 1080):
        self.size = (int(width), int(height))
        self.last = None

    def show(self, frame: np.ndarray) -> None:
        self.last = np.asarray(frame)

    def close(self) -> None:
        pass


class FramebufferDisplay:
    """Writes to a Linux framebuffer device (16, 24 or 32 bpp)."""

    def __init__(self, index: int = 0):
        import fcntl
        import mmap
        self.dev = f"/dev/fb{int(index)}"
        self.fd = os.open(self.dev, os.O_RDWR)
        try:
            var = bytearray(_VSCREENINFO_LEN)
            fcntl.ioctl(self.fd, FB_IOGET_VSCREENINFO, var)
            xres, yres, xvirt, _yvirt, _xo, _yo, bpp, _grey = struct.unpack("8I", var[:32])
            # Re-activate the mode: with fbcon detached (fbcon=map:1) nobody has
            # programmed the CRTC yet and the output would stay dark.
            try:
                struct.pack_into("I", var, 84, FB_ACTIVATE_FORCE)
                fcntl.ioctl(self.fd, FB_IOPUT_VSCREENINFO, var)
            except OSError:
                pass
            stride = self._sysfs(index, "stride", xvirt * bpp // 8)
        except OSError:
            # No ioctl (unusual driver): fall back to sysfs
            xres, yres = self._sysfs_size(index)
            bpp = self._sysfs(index, "bits_per_pixel", 32)
            stride = self._sysfs(index, "stride", xres * bpp // 8)
        if bpp not in (16, 24, 32):
            raise RuntimeError(f"{self.dev}: unsupported {bpp} bpp")
        self.size = (xres, yres)
        self.bpp, self.stride = bpp, stride
        self.mm = mmap.mmap(self.fd, stride * yres, mmap.MAP_SHARED, mmap.PROT_WRITE)
        self._fcntl = fcntl

    @staticmethod
    def _sysfs(index: int, name: str, default: int) -> int:
        try:
            with open(f"/sys/class/graphics/fb{index}/{name}") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return default

    @staticmethod
    def _sysfs_size(index: int) -> Tuple[int, int]:
        with open(f"/sys/class/graphics/fb{index}/virtual_size") as f:
            w, h = f.read().strip().split(",")
        return int(w), int(h)

    @staticmethod
    def list_devices() -> List[str]:
        if not os.path.isdir("/dev"):
            return []
        return sorted(d for d in os.listdir("/dev") if d.startswith("fb"))

    def pack(self, frame: np.ndarray) -> np.ndarray:
        """Pixels as the device wants them: rows of ``stride`` bytes."""
        rgb = to_rgb(frame)
        h, w = rgb.shape[:2]
        if self.bpp == 32:  # XRGB8888 little endian = B, G, R, X in memory
            px = np.zeros((h, w, 4), np.uint8)
            px[:, :, 0], px[:, :, 1], px[:, :, 2] = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]
            px[:, :, 3] = 255
        elif self.bpp == 24:
            px = rgb[:, :, ::-1]
        else:  # RGB565
            r, g, b = (rgb[:, :, i].astype(np.uint16) for i in range(3))
            px = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
            px = px.astype("<u2").view(np.uint8).reshape(h, w * 2)
        px = px.reshape(h, -1)
        if px.shape[1] == self.stride:
            return px
        rows = np.zeros((h, self.stride), np.uint8)
        rows[:, :px.shape[1]] = px
        return rows

    def show(self, frame: np.ndarray) -> None:
        w, h = self.size
        f = np.asarray(frame)
        if f.shape[0] != h or f.shape[1] != w:
            raise ValueError(f"frame {f.shape[1]}x{f.shape[0]} does not match {self.dev} {w}x{h}")
        self.mm.seek(0)
        self.mm.write(self.pack(f).tobytes())
        try:  # best effort: return after the next vertical blank
            self._fcntl.ioctl(self.fd, FBIO_WAITFORVSYNC, struct.pack("I", 0))
        except OSError:
            pass

    def close(self) -> None:
        try:
            self.mm.close()
        finally:
            os.close(self.fd)


class WindowDisplay:
    """Fullscreen window on one monitor, hosted by a pygame-ce subprocess.

    Protocol over the child's stdin/stdout, one text line per message:
      parent -> child  ``frame W H C SEQ`` + W*H*C raw bytes, or ``quit``
      child  -> parent ``ready W H`` once, then ``shown SEQ`` after each flip,
                       ``error MESSAGE`` on failure.
    """

    TIMEOUT_S = 5.0

    def __init__(self, display_index: int = 1, width: Optional[int] = None,
                 height: Optional[int] = None):
        cmd = [sys.executable, os.path.abspath(__file__), "--window", "--display",
               str(int(display_index))]
        if width and height:
            cmd += ["--width", str(int(width)), "--height", str(int(height))]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self._seq = 0
        self._lock = threading.Lock()
        line = self._readline()
        if not line.startswith("ready "):
            raise RuntimeError(f"SLM window did not start: {line or self._stderr()}")
        _, w, h = line.split()
        self.size = (int(w), int(h))

    def _stderr(self) -> str:
        try:
            self.proc.stdin.close()
            return self.proc.communicate(timeout=2)[1].decode(errors="replace").strip()[-800:]
        except Exception:
            return ""

    def _readline(self) -> str:
        box = {}

        def reader():
            box["line"] = self.proc.stdout.readline().decode(errors="replace").strip()
        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(self.TIMEOUT_S)
        return box.get("line", "")

    def show(self, frame: np.ndarray) -> None:
        f = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
        h, w = f.shape[:2]
        c = 1 if f.ndim == 2 else f.shape[2]
        with self._lock:
            self._seq += 1
            seq = self._seq
            if self.proc.poll() is not None:
                raise RuntimeError("SLM window process has exited: " + self._stderr())
            self.proc.stdin.write(f"frame {w} {h} {c} {seq}\n".encode())
            self.proc.stdin.write(f.tobytes())
            self.proc.stdin.flush()
            reply = self._readline()
        if reply != f"shown {seq}":
            raise RuntimeError(f"SLM window did not confirm frame {seq}: {reply or self._stderr()}")

    def close(self) -> None:
        try:
            self.proc.stdin.write(b"quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()

    @staticmethod
    def list_displays() -> list:
        """Desktop sizes of the attached monitors, index = displayIndex."""
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--list"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip()[-400:] or "could not list displays")
        return [[int(v) for v in line.split()] for line in out.stdout.split("\n") if line.strip()]


def open_display(kind: str, width: int, height: int, index: int):
    kind = (kind or "virtual").lower()
    if kind == "virtual":
        return VirtualDisplay(width, height)
    if kind == "framebuffer":
        return FramebufferDisplay(index)
    if kind == "window":
        return WindowDisplay(index, width, height)
    raise ValueError(f"unknown SLM display backend {kind!r} (virtual, framebuffer, window)")


# ───────────────────────── subprocess side (pygame-ce) ─────────────────────────

def _read_exact(stream, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _pygame():
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        import pygame
    except ImportError:
        sys.stderr.write("pygame-ce is not installed: pip install pygame-ce "
                         "(extra imswitch[slm])\n")
        sys.exit(2)
    pygame.display.init()
    return pygame


def _window_main(argv) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--window", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--display", type=int, default=1)
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--height", type=int, default=0)
    a = p.parse_args(argv)
    pygame = _pygame()
    sizes = pygame.display.get_desktop_sizes()
    if a.list:
        print("\n".join(f"{w} {h}" for w, h in sizes))
        return 0
    idx = max(0, min(a.display, len(sizes) - 1))
    w, h = (a.width, a.height) if a.width and a.height else sizes[idx]
    flags = pygame.FULLSCREEN | pygame.NOFRAME
    try:
        screen = pygame.display.set_mode((w, h), flags | pygame.SCALED, display=idx, vsync=1)
    except pygame.error:
        screen = pygame.display.set_mode((w, h), flags, display=idx)
    pygame.display.set_caption("ImSwitch SLM")
    pygame.mouse.set_visible(False)
    screen.fill((0, 0, 0))
    pygame.display.flip()
    sw, sh = screen.get_size()
    sys.stdout.write(f"ready {sw} {sh}\n")
    sys.stdout.flush()
    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline().decode().strip()
        if not line or line == "quit":
            break
        try:
            _, fw, fh, fc, seq = line.split()
            fw, fh, fc = int(fw), int(fh), int(fc)
            raw = _read_exact(stdin, fw * fh * fc)
            arr = np.frombuffer(raw, np.uint8).reshape(fh, fw) if fc == 1 \
                else np.frombuffer(raw, np.uint8).reshape(fh, fw, fc)
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(to_rgb(arr).swapaxes(0, 1)))
            screen.fill((0, 0, 0))
            screen.blit(surf, ((sw - fw) // 2, (sh - fh) // 2))
            pygame.display.flip()
            pygame.event.pump()
            sys.stdout.write(f"shown {seq}\n")
        except Exception as e:  # keep the protocol alive; the parent raises
            sys.stdout.write(f"error {e}\n")
        sys.stdout.flush()
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(_window_main(sys.argv[1:]))
