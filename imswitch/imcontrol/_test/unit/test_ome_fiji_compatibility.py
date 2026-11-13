"""What ImSwitch writes opens as OME in Fiji: Bio-Formats (OME-TIFF) and N5 (OME-Zarr).

Runs Fiji's own jars headlessly; skipped when Fiji or Java is not installed.
Set FIJI_HOME to a Fiji.app folder if it is not in /Applications or ~/Fiji.app.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from imswitch.imcontrol.model.aufgaben.storage import save_image

FIJI = next((Path(p) for p in (os.environ.get("FIJI_HOME"), "/Applications/Fiji.app",
                               str(Path.home() / "Fiji.app")) if p and Path(p, "jars").is_dir()),
            None)
pytestmark = pytest.mark.skipif(FIJI is None or shutil.which("java") is None,
                                reason="needs Fiji (FIJI_HOME) and java")

BF_JARS = ("aircompressor", "commons-io", "commons-lang-", "commons-lang3", "guava", "joda-time",
           "kryo", "logback-classic", "logback-core", "minlog", "native-lib-loader", "objenesis",
           "perf4j", "reflectasm", "slf4j-api", "snakeyaml", "xercesImpl")
N5_JARS = ("n5", "gson", "commons-", "jblosc", "jna", "zstd", "aircompressor", "xz", "lz4",
           "scijava-common", "imglib2", "joml", "ejml", "jama", "Jama")

N5_CHECK = """
import org.janelia.saalfeldlab.n5.*;
import org.janelia.saalfeldlab.n5.zarr.N5ZarrReader;
import org.janelia.saalfeldlab.n5.universe.*;
import java.util.Arrays;
public class N5Check {
  public static void main(String[] a) throws Exception {
    N5ZarrReader r = new N5ZarrReader(a[0]);
    N5TreeNode root = new N5DatasetDiscoverer(r,
        Arrays.asList(N5DatasetDiscoverer.DEFAULT_PARSERS),
        Arrays.asList(N5DatasetDiscoverer.DEFAULT_GROUP_PARSERS)).discoverAndParseRecursive("");
    DatasetAttributes at = r.getDatasetAttributes("0");
    r.readBlock("0", at, new long[at.getNumDimensions()]);
    System.out.println("METADATA " + root.getMetadata().getClass().getSimpleName()
        + " DIMS " + Arrays.toString(at.getDimensions()));
    System.exit(0);
  }
}
"""


def _classpath(prefixes, extra=()):
    jars = sorted(p for p in (FIJI / "jars").glob("*.jar") if p.name.startswith(prefixes))
    return os.pathsep.join([str(j) for j in jars] + list(extra))


def _bioformats(path) -> str:
    cp = _classpath(BF_JARS, [str(FIJI / "jars" / "bio-formats" / "*")])
    return subprocess.run(["java", "-cp", cp, "loci.formats.tools.ImageInfo", "-nopix", "-noflat",
                           str(path)], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          timeout=120).stdout


def test_bioformats_reads_ome_tiff_as_ome(tmp_path):
    rng = np.random.default_rng(0)
    mono = save_image(tmp_path / "mono.ome.tif",
                      rng.integers(0, 4000, (2, 3, 1100, 1300), dtype=np.uint16), "CZYX",
                      pixel_size_um=0.65, z_step_um=2.0, channel_names=["LED", "Laser488"],
                      exposure_ms=[5, 20], origin_um=(100.0, 200.0, 10.0))
    out = _bioformats(mono)
    assert "OMETiffReader initializing" in out, out[-2000:]   # OME, not a plain TIFF
    assert re.search(r"SizeC = 2", out) and re.search(r"SizeZ = 3", out)
    assert re.search(r"Resolutions = 3", out)                  # SubIFD pyramid
    rgb = save_image(tmp_path / "rgb.ome.tif", rng.integers(0, 255, (60, 80, 3), dtype=np.uint8),
                     pixel_size_um=1.0, channel_names=["brightfield"])
    out = _bioformats(rgb)
    assert "OMETiffReader initializing" in out and re.search(r"RGB = true \(3\)", out), out[-2000:]


def test_fiji_n5_reads_ome_zarr(tmp_path):
    store = save_image(tmp_path / "x.ome.zarr",
                       np.arange(2 * 64 * 80, dtype=np.uint16).reshape(2, 64, 80), "CYX",
                       pixel_size_um=0.5, channel_names=["a", "b"])
    source = tmp_path / "N5Check.java"
    source.write_text(N5_CHECK)
    out = subprocess.run(["java", "-cp", _classpath(N5_JARS), str(source), str(store)],
                         capture_output=True, text=True, timeout=180, cwd=tmp_path)
    assert "METADATA OmeNgffMetadata DIMS [80, 64, 1, 2, 1]" in out.stdout, out.stdout + out.stderr
