"""
FileManager API tests for ImSwitch backend.
"""
import os
import shutil
from pathlib import Path
import uuid
import tempfile

import pytest

from imswitch.imcommon.model import dirtools


def _base_data_path() -> Path:
    return Path(dirtools.UserFileDirs.getValidatedDataPath()).resolve()


def _assert_status(response, ok_codes=(200, 201)):
    if response.status_code not in ok_codes:
        pytest.fail(f"Unexpected status {response.status_code}: {response.text}")


def _create_folder(api_server, name: str, parent_id: str = "") -> dict:
    response = api_server.post(
        "/imswitch/api/FileManager/folder",
        json={"name": name, "parentId": parent_id},
    )
    _assert_status(response)
    return response.json()


def _upload_file(api_server, parent_id: str, filename: str, content: bytes) -> dict:
    with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_name = tmp.name
    try:
        with open(tmp_name, "rb") as handle:
            response = api_server.post(
                "/imswitch/api/FileManager/upload",
                data={"parentId": parent_id},
                files={"file": (filename, handle)},
            )
        _assert_status(response)
        return response.json()
    finally:
        os.unlink(tmp_name)


@pytest.fixture(scope="function")
def filemanager_root():
    base_path = _base_data_path()
    root_name = f"_test_filemanager_{uuid.uuid4().hex}"
    root_path = base_path / root_name
    try:
        yield root_name
    finally:
        if root_path.exists():
            shutil.rmtree(root_path, ignore_errors=True)


def test_filemanager_flow(api_server, filemanager_root):
    base_path = _base_data_path()

    # Create base folder via API
    root_rel = _create_folder(api_server, filemanager_root)["path"]

    # Create destination folder under root
    dest_rel = _create_folder(api_server, "dest", root_rel)["path"]

    # Upload a file into root
    uploaded_rel = _upload_file(api_server, root_rel, "example.txt", b"hello world")["path"]

    # Copy into same folder -> should auto-rename
    response = api_server.post(
        "/imswitch/api/FileManager/copy",
        json={"sourceIds": [uploaded_rel], "destinationId": root_rel},
    )
    assert response.status_code == 200
    copied_paths = response.json()["destinations"]
    assert len(copied_paths) == 1
    copied_rel = copied_paths[0]
    assert copied_rel != uploaded_rel
    assert copied_rel.startswith(root_rel)

    # Move copied file to dest folder
    response = api_server.put(
        "/imswitch/api/FileManager/move",
        json={"sourceIds": [copied_rel], "destinationId": dest_rel},
    )
    assert response.status_code == 200
    moved_rel = response.json()["destinations"][0]
    assert moved_rel.startswith(dest_rel)

    # Rename moved file
    new_name = "renamed.txt"
    response = api_server.patch(
        "/imswitch/api/FileManager/rename",
        json={"id": moved_rel, "newName": new_name},
    )
    assert response.status_code == 200

    # Delete renamed file
    response = api_server.delete(
        "/imswitch/api/FileManager",
        json={"ids": [str(Path(dest_rel) / new_name)]},
    )
    assert response.status_code == 200

    # Verify delete on disk
    deleted_path = base_path / dest_rel.lstrip("/") / new_name
    assert not deleted_path.exists()


def test_filemanager_path_traversal_blocked(api_server, filemanager_root):
    # Invalid name with separator
    response = api_server.post(
        "/imswitch/api/FileManager/folder",
        json={"name": "bad/name", "parentId": ""},
    )
    assert response.status_code == 400

    # Invalid parent path traversal
    response = api_server.post(
        "/imswitch/api/FileManager/folder",
        json={"name": "safe", "parentId": "../"},
    )
    assert response.status_code == 400


def test_filemanager_copy_move_between_folders(api_server, filemanager_root):
    base_path = _base_data_path()
    root_rel = _create_folder(api_server, filemanager_root)["path"]
    dest_a = _create_folder(api_server, "dest_a", root_rel)["path"]
    dest_b = _create_folder(api_server, "dest_b", root_rel)["path"]

    uploaded_rel = _upload_file(api_server, dest_a, "copyme.txt", b"copy me")["path"]

    response = api_server.post(
        "/imswitch/api/FileManager/copy",
        json={"sourceIds": [uploaded_rel], "destinationId": dest_b},
    )
    assert response.status_code == 200
    copied_rel = response.json()["destinations"][0]
    assert copied_rel.startswith(dest_b)
    assert (base_path / copied_rel.lstrip("/")).exists()

    response = api_server.put(
        "/imswitch/api/FileManager/move",
        json={"sourceIds": [uploaded_rel], "destinationId": dest_b},
    )
    assert response.status_code == 200
    moved_rel = response.json()["destinations"][0]
    assert moved_rel.startswith(dest_b)
    assert not (base_path / uploaded_rel.lstrip("/")).exists()
    assert (base_path / moved_rel.lstrip("/")).exists()


def test_filemanager_upload_rejects_invalid_name(api_server, filemanager_root):
    root_rel = _create_folder(api_server, filemanager_root)["path"]
    with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
        tmp.write(b"invalid")
        tmp.flush()
        tmp_name = tmp.name
    try:
        with open(tmp_name, "rb") as handle:
            response = api_server.post(
                "/imswitch/api/FileManager/upload",
                data={"parentId": root_rel},
                files={"file": ("bad/name.txt", handle)},
            )
        assert response.status_code == 400
    finally:
        os.unlink(tmp_name)


def test_filemanager_download_and_preview(api_server, filemanager_root):
    root_rel = _create_folder(api_server, filemanager_root)["path"]
    uploaded_rel = _upload_file(api_server, root_rel, "preview.txt", b"preview-data")["path"]

    response = api_server.get(f"/imswitch/api/FileManager/download{uploaded_rel}")
    assert response.status_code == 200
    assert b"preview-data" in response.content

    response = api_server.get(f"/imswitch/api/FileManager/preview{uploaded_rel}")
    assert response.status_code == 200
    assert b"preview-data" in response.content
