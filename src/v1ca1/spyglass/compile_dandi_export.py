"""Compile and upload a validated Spyglass paper export to DANDI."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
from typing import Any


DEFAULT_PAPER_ID = "kyu_v1ca1"
DEFAULT_DANDISET_ID = "001958"
DEFAULT_UPLOAD_PROCESSES = 4
DEFAULT_VALIDATE_PROCESSES = 8


def _decode_hdf5_strings(value: Any) -> list[str]:
    """Return scalar or array-like HDF5 string data as Python strings."""
    if hasattr(value, "reshape"):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    return [item.decode() if isinstance(item, bytes) else str(item) for item in values]


def _external_media_paths(nwb_path: Path) -> tuple[Path, ...]:
    """Read local video paths from every ImageSeries in one NWB file."""
    import h5py

    from dandi.consts import VIDEO_FILE_EXTENSIONS

    paths: list[Path] = []
    with h5py.File(nwb_path, "r") as nwb_file:

        def collect(_name: str, item: Any) -> None:
            if not isinstance(item, h5py.Group):
                return
            neurodata_type = item.attrs.get("neurodata_type")
            if isinstance(neurodata_type, bytes):
                neurodata_type = neurodata_type.decode()
            if neurodata_type != "ImageSeries" or "external_file" not in item:
                return
            for value in _decode_hdf5_strings(item["external_file"][()]):
                path = Path(value)
                if path.suffix.lower() in VIDEO_FILE_EXTENSIONS:
                    paths.append(path)

        nwb_file.visititems(collect)
    return tuple(dict.fromkeys(paths))


def _has_hdf5_external_links(nwb_path: Any) -> bool:
    """Check HDF5 links without failing on unrelated dangling soft links."""
    import h5py

    from dandi.pynwb_utils import open_readable

    with open_readable(nwb_path) as readable, h5py.File(readable, "r") as nwb_file:
        visited = set()

        def visit(group: Any) -> bool:
            visited.add(hash(group.id))
            for name in group.keys():
                link = group.get(name, getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    return True
                try:
                    item = group.get(name)
                except KeyError:
                    continue
                if (
                    isinstance(item, h5py.Group)
                    and hash(item.id) not in visited
                    and visit(item)
                ):
                    return True
            return False

        return visit(nwb_file)


def _stage_external_media(nwb_path: Path, destination_dir: Path) -> int:
    """Copy relative video references beside a staged NWB file."""
    staged = 0
    destination_root = destination_dir.resolve()
    for media_path in _external_media_paths(nwb_path):
        if media_path.is_absolute():
            if not media_path.is_file():
                raise FileNotFoundError(media_path)
            continue
        if "://" in str(media_path):
            continue

        source = nwb_path.parent / media_path
        destination = destination_dir / media_path
        try:
            destination.resolve().relative_to(destination_root)
        except ValueError as error:
            raise ValueError(
                f"External media path escapes the staging directory: {media_path}"
            ) from error
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(destination):
            if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
                raise FileExistsError(
                    f"Conflicting staged media file: {destination}"
                )
            continue
        shutil.copy2(source, destination)
        staged += 1
    return staged


def _nested_image_series(nwb: Any) -> list[dict[str, Any]]:
    """Collect external files from ImageSeries at any NWB container depth."""
    import pynwb

    from dandi import pynwb_utils
    from dandi.consts import VIDEO_FILE_EXTENSIONS

    image_series = []
    seen = set()
    for item in nwb.objects.values():
        if not isinstance(item, pynwb.image.ImageSeries) or item.object_id in seen:
            continue
        seen.add(item.object_id)
        if item.external_file is None:
            continue
        external_files = []
        for value in item.external_file:
            path = Path(value)
            if path.suffix.lower() in VIDEO_FILE_EXTENSIONS:
                external_files.append(path)
            else:
                pynwb_utils.lgr.warning(
                    "external file %s should be one of: %s",
                    value,
                    ", ".join(VIDEO_FILE_EXTENSIONS),
                )
        image_series.append(
            {
                "id": item.object_id,
                "name": item.name,
                "external_files": external_files,
            }
        )
    return image_series


def _external_path_rewrite_plan(
    nwb_path: Path,
) -> list[tuple[str, int, str, str]]:
    """Plan nested ImageSeries rewrites for one organized NWB file."""
    import h5py

    from dandi.consts import VIDEO_FILE_EXTENSIONS

    rewrites: list[tuple[str, int, str, str]] = []
    with h5py.File(nwb_path, "r") as nwb_file:
        image_series = []

        def collect(name: str, item: Any) -> None:
            if not isinstance(item, h5py.Group):
                return
            neurodata_type = item.attrs.get("neurodata_type")
            if isinstance(neurodata_type, bytes):
                neurodata_type = neurodata_type.decode()
            if neurodata_type == "ImageSeries" and "external_file" in item:
                image_series.append((name, item.attrs.get("object_id")))

        nwb_file.visititems(collect)
        for group_name, object_id in image_series:
            if isinstance(object_id, bytes):
                object_id = object_id.decode()
            if not object_id:
                raise ValueError(f"ImageSeries lacks object_id in {nwb_path}.")
            dataset = nwb_file[f"{group_name}/external_file"]
            if len(dataset.shape) != 1:
                raise ValueError(
                    f"Expected one-dimensional external_file in {nwb_path}."
                )
            for index, current in enumerate(_decode_hdf5_strings(dataset[()])):
                suffix = Path(current).suffix.lower()
                if suffix not in VIDEO_FILE_EXTENSIONS:
                    continue
                renamed = (
                    Path(nwb_path.stem)
                    / f"{object_id}_external_file_{index}{suffix}"
                ).as_posix()
                target = nwb_path.parent / renamed
                if not target.is_file():
                    raise FileNotFoundError(target)
                rewrites.append(
                    (f"{group_name}/external_file", index, current, renamed)
                )
    return rewrites


def _rewrite_nested_external_file_paths(nwb_path: Path) -> int:
    """Rewrite nested ImageSeries paths in one organized NWB file in place."""
    import h5py

    rewrites = _external_path_rewrite_plan(nwb_path)
    changed = 0
    with h5py.File(nwb_path, "r+") as nwb_file:
        for dataset_path, index, planned_current, renamed in rewrites:
            dataset = nwb_file[dataset_path]
            current = _decode_hdf5_strings(dataset[()])[index]
            if current == renamed:
                continue
            if current != planned_current:
                raise RuntimeError(
                    f"External path changed while repairing {nwb_path}: {current}"
                )
            dataset[index] = renamed
            changed += 1
        nwb_file.flush()

    remaining = [
        (dataset_path, index, current, renamed)
        for dataset_path, index, current, renamed in _external_path_rewrite_plan(
            nwb_path
        )
        if current != renamed
    ]
    if remaining:
        raise RuntimeError(f"Failed to rewrite all external paths in {nwb_path}.")
    return changed


def _organized_nwbs_with_media(dandiset_dir: Path) -> tuple[Path, ...]:
    """Find organized NWBs with companion external-media directories."""
    return tuple(
        sorted(
            path
            for path in dandiset_dir.rglob("*.nwb")
            if path.with_suffix("").is_dir()
        )
    )


def _repair_organized_external_paths(dandiset_dir: Path) -> tuple[Path, ...]:
    """Repair nested ImageSeries paths throughout one organized Dandiset."""
    nwb_paths = _organized_nwbs_with_media(dandiset_dir)
    for nwb_path in nwb_paths:
        changed = _rewrite_nested_external_file_paths(nwb_path)
        print(
            f"Rewrote {changed} external media paths in "
            f"{nwb_path.relative_to(dandiset_dir)}.",
            flush=True,
        )
    return nwb_paths


def _lookup_nwb_translations(source_dir: str, dandiset_dir: str) -> dict[str, str]:
    """Map staged NWB basenames to their organized DANDI paths."""
    import h5py

    object_paths = {}
    for dandi_file in Path(dandiset_dir).rglob("*.nwb"):
        with h5py.File(dandi_file, "r") as nwb_file:
            object_paths[nwb_file.attrs["object_id"]] = dandi_file.relative_to(
                dandiset_dir
            ).as_posix()

    translations = {}
    for source_file in Path(source_dir).glob("*.nwb"):
        with h5py.File(source_file, "r") as nwb_file:
            translations[source_file.name] = object_paths[
                nwb_file.attrs["object_id"]
            ]
    return translations


@contextmanager
def _dandi_export_compatibility() -> Iterator[None]:
    """Support nested videos while using the Spyglass-pinned DANDI client."""
    from dandi import pynwb_utils
    from dandi import organize as dandi_organize
    from dandi.metadata import nwb as dandi_metadata_nwb
    from spyglass.common import common_dandi

    original_get_image_series = pynwb_utils._get_image_series
    original_pynwb_link_check = pynwb_utils.nwb_has_external_links
    original_metadata_link_check = dandi_metadata_nwb.nwb_has_external_links
    original_spyglass_link_check = common_dandi.nwb_has_external_links
    original_organize = dandi_organize.organize
    original_make_file = common_dandi._make_file_in_dandi_dir
    original_lookup = common_dandi.lookup_dandi_translation

    def make_file(file: str, destination_dir: str, skip_raw_files: bool) -> None:
        source = Path(file)
        destination = Path(destination_dir) / source.name
        if os.path.lexists(destination):
            return
        if skip_raw_files and common_dandi.raw_dir in file:
            return
        if _has_hdf5_external_links(source):
            shutil.copy(source, destination)
        else:
            os.symlink(source, destination)
        media_count = _stage_external_media(Path(file), Path(destination_dir))
        if media_count:
            print(
                f"Staged {media_count} external media files for {Path(file).name}.",
                flush=True,
            )

    def organize(*args: Any, **kwargs: Any) -> None:
        original_organize(*args, **kwargs)
        dandiset_dir = Path(
            args[1] if len(args) > 1 else kwargs["dandiset_path"]
        )
        _repair_organized_external_paths(dandiset_dir)

    pynwb_utils._get_image_series = _nested_image_series
    pynwb_utils.nwb_has_external_links = _has_hdf5_external_links
    dandi_metadata_nwb.nwb_has_external_links = _has_hdf5_external_links
    common_dandi.nwb_has_external_links = _has_hdf5_external_links
    dandi_organize.organize = organize
    common_dandi._make_file_in_dandi_dir = make_file
    common_dandi.lookup_dandi_translation = _lookup_nwb_translations
    try:
        yield
    finally:
        pynwb_utils._get_image_series = original_get_image_series
        pynwb_utils.nwb_has_external_links = original_pynwb_link_check
        dandi_metadata_nwb.nwb_has_external_links = original_metadata_link_check
        common_dandi.nwb_has_external_links = original_spyglass_link_check
        dandi_organize.organize = original_organize
        common_dandi._make_file_in_dandi_dir = original_make_file
        common_dandi.lookup_dandi_translation = original_lookup


def _check_remote_metadata(dandiset_id: str) -> dict[str, Any]:
    """Confirm that the authenticated draft has required descriptive metadata."""
    from dandi.dandiapi import DandiAPIClient

    api_key = os.environ.get("DANDI_API_KEY")
    if not api_key:
        raise RuntimeError("DANDI_API_KEY is not set.")
    with DandiAPIClient.for_dandi_instance("dandi") as client:
        client.authenticate(api_key)
        remote = client.get_dandiset(str(dandiset_id), "draft")
        metadata = remote.get_raw_metadata()
    if not metadata.get("description"):
        raise RuntimeError("The draft Dandiset description is empty.")
    if not metadata.get("license"):
        raise RuntimeError("The draft Dandiset license is empty.")
    return metadata


def _preflight(paper_id: str, dandiset_id: str) -> tuple[dict[str, Any], int]:
    """Check export validation, database state, metadata, and staging paths."""
    from spyglass.common.common_dandi import DandiPath, DandiValidation
    from spyglass.common.common_usage import Export
    from spyglass.settings import export_dir

    paper_key = {"paper_id": str(paper_id)}
    if len(Export & paper_key) != 1:
        raise ValueError("paper_id must correspond to exactly one paper export.")
    export_key = (Export & paper_key).fetch1("KEY")
    file_count = len(Export.File & export_key)
    validated_count = len(DandiValidation & export_key)
    violation_count = len(DandiValidation.Violations & export_key)
    if validated_count != file_count:
        raise RuntimeError(
            f"Only {validated_count} of {file_count} exported files were validated."
        )
    if violation_count:
        raise RuntimeError(f"DANDI validation has {violation_count} violations.")
    if DandiPath & export_key:
        raise RuntimeError("DandiPath already contains rows for this export.")

    paper_dir = Path(export_dir) / paper_id
    staging_paths = (
        paper_dir / f"dandiset_{paper_id}",
        paper_dir / str(dandiset_id),
    )
    existing = [path for path in staging_paths if os.path.lexists(path)]
    if existing:
        raise RuntimeError(
            "DANDI staging paths already exist: " + ", ".join(map(str, existing))
        )

    metadata = _check_remote_metadata(dandiset_id)
    print(
        f"Preflight passed: {file_count} validated NWB files; "
        f"Dandiset {dandiset_id} has a description and "
        f"license {metadata['license']}.",
        flush=True,
    )
    return export_key, file_count


def compile_dandi_export(
    *,
    paper_id: str = DEFAULT_PAPER_ID,
    dandiset_id: str = DEFAULT_DANDISET_ID,
    upload_processes: int = DEFAULT_UPLOAD_PROCESSES,
    validate_processes: int = DEFAULT_VALIDATE_PROCESSES,
    preflight_only: bool = False,
) -> int:
    """Compile and upload one validated paper export, returning its file count."""
    if upload_processes < 1 or validate_processes < 1:
        raise ValueError("Process counts must be at least 1.")
    os.environ.setdefault("DANDI_CACHE", "ignore")
    export_key, file_count = _preflight(paper_id, dandiset_id)
    if preflight_only:
        return file_count

    from spyglass.common.common_dandi import DandiPath

    print(
        f"Compiling and uploading {file_count} NWB files to Dandiset "
        f"{dandiset_id}.",
        flush=True,
    )
    with _dandi_export_compatibility():
        DandiPath().compile_dandiset(
            key=export_key,
            dandiset_id=str(dandiset_id),
            dandi_instance="dandi",
            n_compile_processes=1,
            n_upload_processes=upload_processes,
            n_organize_processes=1,
            n_validate_processes=validate_processes,
        )

    inserted_count = len(DandiPath & export_key)
    if inserted_count != file_count:
        raise RuntimeError(
            f"DandiPath contains {inserted_count} of {file_count} expected rows."
        )
    print(f"Upload complete; inserted {inserted_count} DandiPath rows.", flush=True)
    return file_count


def _validate_organized_nwbs(
    nwb_paths: Sequence[Path], dandiset_dir: Path
) -> None:
    """Run the upload-time DANDI validation on selected organized NWBs."""
    from dandi.files import find_dandi_files
    from dandi.validate_types import Severity

    targets = {str(path) for path in nwb_paths}
    dandi_files = {
        str(item.filepath): item for item in find_dandi_files(str(dandiset_dir))
    }
    missing = sorted(targets - dandi_files.keys())
    if missing:
        raise RuntimeError("DANDI did not discover: " + ", ".join(missing))

    blocking = []
    for path in nwb_paths:
        print(f"Validating {path.relative_to(dandiset_dir)}.", flush=True)
        errors = dandi_files[str(path)].get_validation_errors()
        blocking.extend(
            (path, error)
            for error in errors
            if error.severity is not None and error.severity >= Severity.ERROR
        )
    if blocking:
        details = "\n".join(
            f"{path}: {error.id}: {error.message}" for path, error in blocking
        )
        raise RuntimeError(f"Organized raw NWB validation failed:\n{details}")
    print(f"Validated {len(nwb_paths)} organized raw NWBs.", flush=True)


def _expected_local_asset_paths(dandiset_dir: Path) -> set[str]:
    """List local NWB and video asset paths relative to a Dandiset root."""
    from dandi.consts import VIDEO_FILE_EXTENSIONS

    extensions = {".nwb", *VIDEO_FILE_EXTENSIONS}
    return {
        path.relative_to(dandiset_dir).as_posix()
        for path in dandiset_dir.rglob("*")
        if (path.is_file() or path.is_symlink())
        and path.suffix.lower() in extensions
    }


def _verify_remote_assets(dandiset_id: str, dandiset_dir: Path) -> int:
    """Verify that every organized NWB and video exists in the draft Dandiset."""
    from dandi.dandiapi import DandiAPIClient

    api_key = os.environ.get("DANDI_API_KEY")
    if not api_key:
        raise RuntimeError("DANDI_API_KEY is not set.")
    with DandiAPIClient.for_dandi_instance("dandi") as client:
        client.authenticate(api_key)
        remote = client.get_dandiset(str(dandiset_id), "draft")
        remote_paths = {asset.path for asset in remote.get_assets()}
    expected_paths = _expected_local_asset_paths(dandiset_dir)
    missing = sorted(expected_paths - remote_paths)
    if missing:
        raise RuntimeError(
            "Draft Dandiset is missing assets: " + ", ".join(missing)
        )
    print(f"Verified {len(expected_paths)} assets on DANDI.", flush=True)
    return len(expected_paths)


def _insert_dandi_path_rows(
    *,
    export_key: dict[str, Any],
    dandiset_id: str,
    source_dir: Path,
    dandiset_dir: Path,
) -> int:
    """Insert local-to-DANDI NWB translations after successful upload."""
    from spyglass.common.common_dandi import DandiPath
    from spyglass.common.common_usage import Export

    file_count = len(Export.File & export_key)
    existing_count = len(DandiPath & export_key)
    if existing_count == file_count:
        return existing_count
    if existing_count:
        raise RuntimeError(
            f"DandiPath contains a partial set of {existing_count} rows."
        )

    translations = _lookup_nwb_translations(str(source_dir), str(dandiset_dir))
    if len(translations) != file_count:
        raise RuntimeError(
            f"Found {len(translations)} of {file_count} NWB path translations."
        )
    rows = [
        {
            **(
                Export.File()
                & export_key
                & f"file_path LIKE '%{local_name}'"
            ).fetch1(),
            "filename": local_name,
            "dandi_path": dandi_path,
            "dandiset_id": str(dandiset_id),
            "dandi_instance": "dandi",
        }
        for local_name, dandi_path in translations.items()
    ]
    DandiPath().insert(rows, ignore_extra_fields=True)
    return len(DandiPath & export_key)


def resume_dandi_export(
    *,
    paper_id: str = DEFAULT_PAPER_ID,
    dandiset_id: str = DEFAULT_DANDISET_ID,
    upload_processes: int = DEFAULT_UPLOAD_PROCESSES,
) -> int:
    """Repair and resume a compiled DANDI export after an upload failure."""
    if upload_processes < 1:
        raise ValueError("upload_processes must be at least 1.")
    os.environ.setdefault("DANDI_CACHE", "ignore")

    from dandi import upload as dandi_upload
    from spyglass.common.common_dandi import DandiPath
    from spyglass.common.common_usage import Export
    from spyglass.settings import export_dir

    paper_key = {"paper_id": str(paper_id)}
    if len(Export & paper_key) != 1:
        raise ValueError("paper_id must correspond to exactly one paper export.")
    export_key = (Export & paper_key).fetch1("KEY")
    file_count = len(Export.File & export_key)
    existing_count = len(DandiPath & export_key)
    if existing_count not in (0, file_count):
        raise RuntimeError(
            f"DandiPath contains a partial set of {existing_count} rows."
        )

    paper_dir = Path(export_dir) / paper_id
    source_dir = paper_dir / f"dandiset_{paper_id}"
    dandiset_dir = paper_dir / str(dandiset_id)
    for path in source_dir, dandiset_dir:
        if not path.is_dir():
            raise FileNotFoundError(path)
    _check_remote_metadata(dandiset_id)

    with _dandi_export_compatibility():
        raw_nwb_paths = _repair_organized_external_paths(dandiset_dir)
        if not raw_nwb_paths:
            raise RuntimeError("No organized NWBs with external media were found.")
        _validate_organized_nwbs(raw_nwb_paths, dandiset_dir)
        print(f"Resuming upload of {len(raw_nwb_paths)} raw NWBs.", flush=True)
        dandi_upload.upload(
            [str(path) for path in raw_nwb_paths],
            dandi_instance="dandi",
            jobs=upload_processes,
        )
        _verify_remote_assets(dandiset_id, dandiset_dir)
        inserted_count = _insert_dandi_path_rows(
            export_key=export_key,
            dandiset_id=dandiset_id,
            source_dir=source_dir,
            dandiset_dir=dandiset_dir,
        )

    if inserted_count != file_count:
        raise RuntimeError(
            f"DandiPath contains {inserted_count} of {file_count} expected rows."
        )
    print(f"Resume complete; inserted {inserted_count} DandiPath rows.", flush=True)
    return inserted_count


def _parser() -> argparse.ArgumentParser:
    """Build the DANDI compilation command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", default=DEFAULT_PAPER_ID)
    parser.add_argument("--dandiset-id", default=DEFAULT_DANDISET_ID)
    parser.add_argument(
        "--upload-processes", type=int, default=DEFAULT_UPLOAD_PROCESSES
    )
    parser.add_argument(
        "--validate-processes", type=int, default=DEFAULT_VALIDATE_PROCESSES
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Spyglass-to-DANDI compilation workflow."""
    args = _parser().parse_args(argv)
    if args.resume:
        if args.preflight_only:
            raise ValueError("--resume and --preflight-only cannot be combined.")
        resume_dandi_export(
            paper_id=args.paper_id,
            dandiset_id=args.dandiset_id,
            upload_processes=args.upload_processes,
        )
    else:
        compile_dandi_export(
            paper_id=args.paper_id,
            dandiset_id=args.dandiset_id,
            upload_processes=args.upload_processes,
            validate_processes=args.validate_processes,
            preflight_only=args.preflight_only,
        )


if __name__ == "__main__":
    main()


__all__ = ["compile_dandi_export", "main", "resume_dandi_export"]
