"""The OME-Zarr viewer lists datasets with find_datasets: it must not walk into zarr stores."""
from pathlib import Path

from imswitch.imcontrol.controller.server.ImSwitchServer import find_datasets


def test_lists_zarr_stores_and_ome_tiffs_without_entering_stores(tmp_path: Path):
    run = tmp_path / "Aufgaben" / "20261118_214930_job"
    (run / "ra_0000_img.ome.zarr" / "0" / "0").mkdir(parents=True)
    (run / "ra_0000_img.ome.zarr" / "nested.zarr").mkdir()     # inside a store: not listed
    (run / "ra_0000_img.ome.tif").write_bytes(b"")
    (run / "job.json").write_text("{}")
    (tmp_path / ".hidden" / "x.zarr").mkdir(parents=True)

    items = find_datasets(tmp_path, tmp_path)

    assert sorted(i["path"] for i in items) == [
        "/Aufgaben/20261118_214930_job/ra_0000_img.ome.tif",
        "/Aufgaben/20261118_214930_job/ra_0000_img.ome.zarr",
    ]
    assert {i["folder"] for i in items} == {"/Aufgaben/20261118_214930_job"}
    assert {i["format"] for i in items} == {"zarr", "tiff"}
