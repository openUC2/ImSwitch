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


def test_file_list_shows_a_zarr_store_as_one_folder(tmp_path: Path):
    from imswitch.imcontrol.controller.server.ImSwitchServer import list_items
    store = tmp_path / "run" / "img.ome.zarr"
    (store / "0" / "0").mkdir(parents=True)
    (store / "0" / "0" / "0").write_bytes(b"")
    (tmp_path / "run" / "job.json").write_text("{}")

    assert sorted(i["path"] for i in list_items(tmp_path, tmp_path)) == [
        "/run", "/run/img.ome.zarr", "/run/job.json"]
    # asked for the store itself, its content is listed
    assert "/run/img.ome.zarr/0/0/0" in [i["path"] for i in list_items(store, tmp_path)]


def test_data_mount_follows_a_changed_data_folder(tmp_path: Path, monkeypatch):
    from imswitch.imcommon.model import dirtools
    from imswitch.imcontrol.controller.server.ImSwitchServer import _DataStaticFiles
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (second / "new.zarr").mkdir()
    current = [str(first)]
    monkeypatch.setattr(dirtools.UserFileDirs, "getValidatedDataPath", lambda: current[0])
    mount = _DataStaticFiles(directory=str(first))

    assert mount.lookup_path("new.zarr")[1] is None
    current[0] = str(second)
    assert mount.lookup_path("new.zarr")[1] is not None
