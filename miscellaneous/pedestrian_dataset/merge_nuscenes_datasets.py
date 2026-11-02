"""
Standalone tool that merges two or more ALREADY-GENERATED nuScenes-format
datasets into a single nuScenes-format dataset.

Info:
  - one nuScenes `log` per input log;
  - dataset-wide tables (sensor, category, attribute, visibility) shared
    once across the whole merged output;
  - per-log tables (log, map, calibrated_sensor, scene, sample, ego_pose,
    sample_data, sample_annotation, instance) simply pooled together;
  - every distinct map image is copied

Assumptions:
  - All input datasets were produced for the same vehicle / channel map.
  - Scene names (and therefore the Stage-1 folder basenames they were
    derived from) must be unique across every input being merged.
  - Instances/annotations are scoped per log/scene and are never linked
    across logs.
  - sample_data filenames are unique dataset-wide once scene names are
    unique - they are copied verbatim, never renamed.

Usage:
    python merge_nuscenes_datasets.py \
        --inputs /path/to/dataset_A /path/to/dataset_B /path/to/dataset_C \
        --output /path/to/merged_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import uuid
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("merge_nuscenes_datasets")

# ---------------------------------------------------------------------------
# Constants (mirroring the layout produced by converters/nuscenes_converter.py)
# ---------------------------------------------------------------------------

_META_DIR = "v1.0-trainval"

_GLOBAL_TABLE_NAMES = ("sensor", "category", "attribute", "visibility")
_PER_LOG_TABLE_NAMES = (
    "log", "map", "calibrated_sensor", "scene", "sample", "ego_pose",
    "sample_data", "sample_annotation", "instance",
)
_TABLE_NAMES = _GLOBAL_TABLE_NAMES + _PER_LOG_TABLE_NAMES


def _token() -> str:
    """Generate a 32-character hex token (UUID4 without hyphens)."""
    return uuid.uuid4().hex


def _sanitize(name: str) -> str:
    """Make a string safe to use as a filename component."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _meta_dir(nusc_root: str) -> str:
    return os.path.join(nusc_root, _META_DIR, _META_DIR)


def _load_tables(nusc_root: str) -> Dict[str, list]:
    """
    Load every nuScenes JSON table for one input dataset.
    Raises FileNotFoundError with a clear message if the expected layout
    (<root>/v1.0-trainval/v1.0-trainval/*.json) is not found.
    """
    meta_dir = _meta_dir(nusc_root)
    if not os.path.isdir(meta_dir):
        raise FileNotFoundError(
            f"'{nusc_root}' does not look like a nuScenes dataset "
            f"(expected metadata directory not found: {meta_dir})"
        )

    tables: Dict[str, list] = {}
    for name in _TABLE_NAMES:
        path = os.path.join(meta_dir, f"{name}.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing expected table: {path}")
        with open(path) as fh:
            tables[name] = json.load(fh)
    return tables


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_vehicle_invariance(all_tables: List[Dict[str, list]], inputs: List[str]) -> None:
    """
    Every log, across every input dataset, must report the same vehicle -
    this mirrors the single-channel-map assumption the original converter
    relies on (one shared sensor/category/attribute/visibility set for the
    whole run).
    """
    vehicles = set()
    for tables, root in zip(all_tables, inputs):
        for log in tables["log"]:
            vehicles.add(log.get("vehicle"))
    if len(vehicles) > 1:
        raise ValueError(
            "Cannot merge datasets built for different vehicles/channel maps: "
            f"found vehicle values {sorted(v for v in vehicles if v is not None)}. "
            "All merged datasets must originate from the same vehicle."
        )


def _validate_unique_scene_names(all_tables: List[Dict[str, list]], inputs: List[str]) -> None:
    """
    Scene names must be unique across every input being merged - this is
    the exact cross-dataset collision-avoidance rule the original
    converter enforces on Stage-1 folder basenames (scene names are
    derived from them 1:1).
    """
    seen: Dict[str, str] = {}
    dupes: List[str] = []
    for tables, root in zip(all_tables, inputs):
        for scene in tables["scene"]:
            name = scene["name"]
            if name in seen:
                dupes.append(f"'{name}' ({seen[name]} vs {root})")
            else:
                seen[name] = root
    if dupes:
        raise ValueError(
            "Duplicate scene name(s) across the datasets being merged: "
            + "; ".join(dupes)
            + ". Each input dataset must contain globally-unique scene names."
        )


def _global_signature_sensor(row: dict) -> tuple:
    return (row["channel"], row["modality"])


def _global_signature_category(row: dict) -> tuple:
    return (row["name"], row.get("description"), row.get("index"))


def _global_signature_attribute(row: dict) -> tuple:
    return (row["name"], row.get("description"))


def _global_signature_visibility(row: dict) -> tuple:
    return (row["level"], row.get("description"))


_GLOBAL_SIGNATURES = {
    "sensor": _global_signature_sensor,
    "category": _global_signature_category,
    "attribute": _global_signature_attribute,
    "visibility": _global_signature_visibility,
}


def _validate_global_tables_match(all_tables: List[Dict[str, list]], inputs: List[str]) -> None:
    """
    sensor / category / attribute / visibility must be content-identical
    (ignoring tokens) across every input dataset, since the original
    converter builds these tables exactly once per run and assumes they
    are valid for every merged log.
    """
    reference_sig = None
    for table_name in _GLOBAL_TABLE_NAMES:
        sig_fn = _GLOBAL_SIGNATURES[table_name]
        first_sig = None
        for tables, root in zip(all_tables, inputs):
            sig = sorted(sig_fn(row) for row in tables[table_name])
            if first_sig is None:
                first_sig = sig
            elif sig != first_sig:
                raise ValueError(
                    f"'{table_name}' table differs across input datasets "
                    f"(mismatch involving '{root}'). All merged datasets must "
                    f"share the same channel map / category / attribute / "
                    f"visibility definitions."
                )


def _validate(all_tables: List[Dict[str, list]], inputs: List[str]) -> None:
    _validate_vehicle_invariance(all_tables, inputs)
    _validate_unique_scene_names(all_tables, inputs)
    _validate_global_tables_match(all_tables, inputs)


# ---------------------------------------------------------------------------
# Global-table reconciliation (Step B)
# ---------------------------------------------------------------------------

def _build_canonical_global_tables(
    all_tables: List[Dict[str, list]],
) -> Tuple[Dict[str, list], List[Dict[str, Dict[str, str]]]]:
    """
    Use the first input dataset's global tables as the canonical copy
    (validated identical to every other input's).  Build, per input
    dataset and per global table, an old-token -> canonical-token map.

    Returns
        canonical_tables: {table_name: rows} (from the first input, unchanged)
        remaps: list (one per input dataset) of {table_name: {old_tok: new_tok}}
    """
    canonical_tables = {name: all_tables[0][name] for name in _GLOBAL_TABLE_NAMES}

    # canonical signature -> canonical token, per table
    canonical_by_sig: Dict[str, Dict[tuple, str]] = {}
    for name in _GLOBAL_TABLE_NAMES:
        sig_fn = _GLOBAL_SIGNATURES[name]
        canonical_by_sig[name] = {
            sig_fn(row): row["token"] for row in canonical_tables[name]
        }

    remaps: List[Dict[str, Dict[str, str]]] = []
    for tables in all_tables:
        per_table_remap: Dict[str, Dict[str, str]] = {}
        for name in _GLOBAL_TABLE_NAMES:
            sig_fn = _GLOBAL_SIGNATURES[name]
            table_remap: Dict[str, str] = {}
            for row in tables[name]:
                sig = sig_fn(row)
                table_remap[row["token"]] = canonical_by_sig[name][sig]
            per_table_remap[name] = table_remap
        remaps.append(per_table_remap)

    return canonical_tables, remaps


# ---------------------------------------------------------------------------
# Per-log token remap (Step C)
# ---------------------------------------------------------------------------

def _remap_tokens(field, remap: Dict[str, str]):
    """Remap a single token, a list of tokens, or leave falsy values untouched."""
    if field in (None, ""):
        return field
    if isinstance(field, list):
        return [remap.get(t, t) for t in field]
    return remap.get(field, field)


def _remap_one_dataset(
    tables: Dict[str, list],
    global_remap: Dict[str, Dict[str, str]],
) -> Dict[str, list]:
    """
    Generate fresh tokens for every row of every per-log table in this
    dataset, and rewrite all cross-references (both to other per-log rows
    in the same dataset, and to the canonical global tables).

    Returns a dict of table_name -> list of remapped rows (per-log tables
    only; global tables are handled separately).
    """
    local_remap: Dict[str, str] = {}
    for name in _PER_LOG_TABLE_NAMES:
        for row in tables[name]:
            local_remap[row["token"]] = _token()

    out: Dict[str, list] = {name: [] for name in _PER_LOG_TABLE_NAMES}

    for row in tables["log"]:
        new_row = dict(row)
        new_row["token"] = local_remap[row["token"]]
        out["log"].append(new_row)

    for row in tables["map"]:
        new_row = dict(row)
        new_row["token"] = local_remap[row["token"]]
        new_row["log_tokens"] = _remap_tokens(row.get("log_tokens", []), local_remap)
        out["map"].append(new_row)

    for row in tables["calibrated_sensor"]:
        new_row = dict(row)
        new_row["token"] = local_remap[row["token"]]
        new_row["sensor_token"] = _remap_tokens(row["sensor_token"], global_remap["sensor"])
        out["calibrated_sensor"].append(new_row)

    for row in tables["scene"]:
        new_row = dict(row)
        new_row["token"] = local_remap[row["token"]]
        new_row["log_token"] = _remap_tokens(row["log_token"], local_remap)
        new_row["first_sample_token"] = _remap_tokens(row["first_sample_token"], local_remap)
        new_row["last_sample_token"] = _remap_tokens(row["last_sample_token"], local_remap)
        out["scene"].append(new_row)

    for row in tables["sample"]:
        new_row = dict(row)
        new_row["token"] = local_remap[row["token"]]
        new_row["scene_token"] = _remap_tokens(row["scene_token"], local_remap)
        new_row["prev"] = _remap_tokens(row["prev"], local_remap)
        new_row["next"] = _remap_tokens(row["next"], local_remap)
        out["sample"].append(new_row)

    for row in tables["ego_pose"]:
        new_row = dict(row)
        new_row["token"] = local_remap[row["token"]]
        out["ego_pose"].append(new_row)

    for row in tables["sample_data"]:
        new_row = dict(row)
        new_row["token"] = local_remap[row["token"]]
        new_row["sample_token"] = _remap_tokens(row["sample_token"], local_remap)
        new_row["ego_pose_token"] = _remap_tokens(row["ego_pose_token"], local_remap)
        new_row["calibrated_sensor_token"] = _remap_tokens(row["calibrated_sensor_token"], local_remap)
        new_row["prev"] = _remap_tokens(row["prev"], local_remap)
        new_row["next"] = _remap_tokens(row["next"], local_remap)
        # filename is left untouched: it already embeds the (validated-unique)
        # scene name, exactly as the original converter relies on.
        out["sample_data"].append(new_row)

    for row in tables["instance"]:
        new_row = dict(row)
        new_row["token"] = local_remap[row["token"]]
        new_row["category_token"] = _remap_tokens(row["category_token"], global_remap["category"])
        new_row["first_annotation_token"] = _remap_tokens(row.get("first_annotation_token", ""), local_remap)
        new_row["last_annotation_token"] = _remap_tokens(row.get("last_annotation_token", ""), local_remap)
        out["instance"].append(new_row)

    for row in tables["sample_annotation"]:
        new_row = dict(row)
        new_row["token"] = local_remap[row["token"]]
        new_row["sample_token"] = _remap_tokens(row["sample_token"], local_remap)
        new_row["instance_token"] = _remap_tokens(row["instance_token"], local_remap)
        new_row["attribute_tokens"] = _remap_tokens(row.get("attribute_tokens", []), global_remap["attribute"])
        new_row["visibility_token"] = _remap_tokens(row.get("visibility_token", ""), global_remap["visibility"])
        new_row["prev"] = _remap_tokens(row["prev"], local_remap)
        new_row["next"] = _remap_tokens(row["next"], local_remap)
        out["sample_annotation"].append(new_row)

    return out


# ---------------------------------------------------------------------------
# Map-file handling (extension: unique filenames per distinct map image)
# ---------------------------------------------------------------------------

def _rewrite_map_filenames(
    per_log_tables: Dict[str, list],
    input_root: str,
    input_index: int,
    output_root: str,
) -> None:
    """
    Copy every map image referenced by this dataset's `map` table into the
    merged output under a new, deterministic, collision-free filename, and
    rewrite the corresponding map.json 'filename' fields in place.
    """
    prefix = f"{input_index:03d}_{_sanitize(os.path.basename(os.path.normpath(input_root)))}"
    out_maps_dir = os.path.join(output_root, _META_DIR, "maps")
    os.makedirs(out_maps_dir, exist_ok=True)

    # original relative filename (within this input) -> new relative filename
    copied: Dict[str, str] = {}

    for row in per_log_tables["map"]:
        orig_rel = row["filename"]  # e.g. "maps/map.png"
        if orig_rel not in copied:
            src = os.path.join(input_root, _META_DIR, orig_rel)
            orig_basename = os.path.basename(orig_rel)
            new_basename = f"{prefix}__{orig_basename}"
            new_rel = os.path.join("maps", new_basename)

            if not os.path.isfile(src):
                raise FileNotFoundError(
                    f"map.json in '{input_root}' references '{orig_rel}', "
                    f"but the file does not exist at '{src}'."
                )
            dst = os.path.join(output_root, _META_DIR, new_rel)
            shutil.copy2(src, dst)
            copied[orig_rel] = new_rel

        row["filename"] = copied[orig_rel]


# ---------------------------------------------------------------------------
# Sample-data media file copy
# ---------------------------------------------------------------------------

def _copy_sample_data_files(
    per_log_tables: Dict[str, list],
    input_root: str,
    output_root: str,
) -> None:
    """
    Copy every sample_data media file referenced by this dataset into the
    merged output, preserving its relative path (filenames are assumed
    unique dataset-wide, per the scene-name-uniqueness validation already
    performed).
    """
    for row in per_log_tables["sample_data"]:
        rel = row["filename"]
        src = os.path.join(input_root, _META_DIR, rel)
        dst = os.path.join(output_root, _META_DIR, rel)

        if not os.path.isfile(src):
            raise FileNotFoundError(
                f"sample_data.json in '{input_root}' references '{rel}', "
                f"but the file does not exist at '{src}'."
            )
        if os.path.exists(dst):
            raise ValueError(
                f"Filename collision while copying sample_data media: "
                f"'{rel}' already exists in the output dataset. This should "
                f"be impossible given unique scene names across inputs - "
                f"aborting rather than silently overwriting."
            )
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Output directory scaffolding
# ---------------------------------------------------------------------------

def _make_output_dirs(output_root: str, canonical_sensor_table: list) -> None:
    """
    Create the nuScenes directory tree for the merged output: the metadata
    directory, and samples/<channel> + sweeps/<channel> for every channel
    in the (shared) sensor table.
    """
    os.makedirs(os.path.join(output_root, _META_DIR, _META_DIR), exist_ok=True)
    os.makedirs(os.path.join(output_root, _META_DIR, "maps"), exist_ok=True)
    for row in canonical_sensor_table:
        ch = row["channel"]
        os.makedirs(os.path.join(output_root, _META_DIR, "samples", ch), exist_ok=True)
        os.makedirs(os.path.join(output_root, _META_DIR, "sweeps", ch), exist_ok=True)


def _write_json(directory: str, filename: str, data: list) -> None:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=4)
    logger.debug("Written: %s (%d entries)", path, len(data))


# ---------------------------------------------------------------------------
# Main merge routine
# ---------------------------------------------------------------------------

def merge_nuscenes_datasets(input_roots: List[str], output_root: str) -> None:
    if len(input_roots) < 2:
        raise ValueError("Provide at least two input datasets to merge.")

    logger.info("Loading %d input dataset(s)...", len(input_roots))
    all_tables = [_load_tables(root) for root in input_roots]

    logger.info("Validating merge assumptions...")
    _validate(all_tables, input_roots)

    logger.info("Reconciling dataset-wide (global) tables...")
    canonical_global_tables, global_remaps = _build_canonical_global_tables(all_tables)

    _make_output_dirs(output_root, canonical_global_tables["sensor"])

    merged_per_log: Dict[str, list] = {name: [] for name in _PER_LOG_TABLE_NAMES}

    for idx, (tables, root) in enumerate(zip(all_tables, input_roots)):
        logger.info("--- Merging input %d/%d: %s ---", idx + 1, len(input_roots), root)

        remapped = _remap_one_dataset(tables, global_remaps[idx])
        _rewrite_map_filenames(remapped, root, idx, output_root)
        _copy_sample_data_files(remapped, root, output_root)

        for name in _PER_LOG_TABLE_NAMES:
            merged_per_log[name].extend(remapped[name])

    logger.info("Writing merged JSON tables...")
    meta_dir = os.path.join(output_root, _META_DIR, _META_DIR)
    _write_json(meta_dir, "sensor.json", canonical_global_tables["sensor"])
    _write_json(meta_dir, "category.json", canonical_global_tables["category"])
    _write_json(meta_dir, "attribute.json", canonical_global_tables["attribute"])
    _write_json(meta_dir, "visibility.json", canonical_global_tables["visibility"])
    for name in _PER_LOG_TABLE_NAMES:
        _write_json(meta_dir, f"{name}.json", merged_per_log[name])

    logger.info(
        "Merge complete: %d input dataset(s) -> %d scene(s), %d sample(s), "
        "%d sample_data entries, written to %s",
        len(input_roots),
        len(merged_per_log["scene"]),
        len(merged_per_log["sample"]),
        len(merged_per_log["sample_data"]),
        output_root,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Merge two or more already-built nuScenes-format datasets "
                    "into a single nuScenes-format dataset, without re-running "
                    "the bag-to-fusion-dataset pipeline."
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True, metavar="NUSCENES_DATASET_DIR",
        help="Paths to two or more already-generated nuScenes dataset "
             "root directories (each containing a v1.0-trainval/ subfolder).",
    )
    parser.add_argument(
        "--output", required=True, metavar="OUTPUT_DIR",
        help="Path to the directory where the merged nuScenes dataset will be written.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    merge_nuscenes_datasets(args.inputs, args.output)


if __name__ == "__main__":
    main()