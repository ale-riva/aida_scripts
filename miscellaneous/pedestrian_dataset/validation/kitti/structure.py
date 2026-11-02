"""
KITTI structural checks (no IR needed).
"""

from __future__ import annotations

import os
from typing import Dict, List, Set

from ..common import loaders
from ..common.results import SectionResult

_REQUIRED_SUBDIRS = ["image_2", "velodyne", "calib", "label_2"]
_EXT = {"image_2": ".png", "velodyne": ".bin", "calib": ".txt", "label_2": ".txt"}


def _ids(kitti_dir: str, sub: str) -> Set[str]:
    d = os.path.join(kitti_dir, "training", sub)
    return {os.path.splitext(f)[0] for f in loaders.list_dir(d, _EXT[sub])}


def _check_layout(kitti_dir: str, r: SectionResult) -> None:
    training = os.path.join(kitti_dir, "training")
    missing = [s for s in _REQUIRED_SUBDIRS if not os.path.isdir(os.path.join(training, s))]
    if missing:
        r.fail("layout", f"missing training/ subdirectories: {missing}")
    else:
        r.ok("layout", "training/{image_2,velodyne,calib,label_2} all present.")
    if os.path.isdir(os.path.join(kitti_dir, "ImageSets")):
        r.ok("imagesets_dir", "ImageSets/ present.")
    else:
        r.fail("imagesets_dir", "ImageSets/ directory missing.")


def _check_bin_sizes(kitti_dir: str, r: SectionResult) -> None:
    vdir = os.path.join(kitti_dir, "training", "velodyne")
    names = loaders.list_dir(vdir, ".bin")
    if not names:
        r.fail("bin_present", "no .bin files in velodyne/.")
        return
    bad_size, dangling = [], []
    for f in names:
        p = os.path.join(vdir, f)
        if os.path.islink(p) and not os.path.exists(p):
            dangling.append(f)
            continue
        try:
            if os.path.getsize(p) % 16 != 0:
                bad_size.append(f)
        except OSError:
            dangling.append(f)
    if dangling:
        r.fail("bin_dangling", f"{len(dangling)} velodyne entry(ies) point to a missing target "
               f"(e.g. {dangling[:3]}).")
    if bad_size:
        r.fail("bin_size", f"{len(bad_size)}/{len(names)} .bin files not a multiple of 16 bytes "
               f"(e.g. {bad_size[:3]}).")
    elif not dangling:
        r.ok("bin_size", f"all {len(names)} .bin files are 16-byte aligned.")


def _check_parity_and_contiguity(kitti_dir: str, r: SectionResult) -> None:
    id_sets: Dict[str, Set[str]] = {s: _ids(kitti_dir, s) for s in _REQUIRED_SUBDIRS}
    ref = id_sets["velodyne"]
    n = len(ref)

    # Parity across folders.
    mism = []
    for s in _REQUIRED_SUBDIRS:
        if id_sets[s] != ref:
            only_ref = sorted(ref - id_sets[s])[:2]
            only_s = sorted(id_sets[s] - ref)[:2]
            mism.append(f"{s} ({len(id_sets[s])}; +{only_s} -{only_ref})")
    if mism:
        r.fail("parity", f"id sets differ from velodyne ({n}): " + "; ".join(mism))
    else:
        r.ok("parity", f"image_2/calib/label_2/velodyne share the same {n} ids.",
             value=float(n))

    # Contiguity 000000..N-1.
    expected = {loaders.frame_id(i) for i in range(n)}
    if ref != expected:
        missing = sorted(expected - ref)[:3]
        extra = sorted(ref - expected)[:3]
        r.fail("contiguity", f"ids are not contiguous 000000..{loaders.frame_id(n-1)} "
               f"(missing {missing}, unexpected {extra}).")
    else:
        r.ok("contiguity", f"ids contiguous 000000..{loaders.frame_id(max(n-1,0))}.")


def _check_splits(kitti_dir: str, r: SectionResult) -> None:
    image_sets = os.path.join(kitti_dir, "ImageSets")
    files = {n: os.path.join(image_sets, f"{n}.txt") for n in ("train", "val", "trainval")}
    present = {n: p for n, p in files.items() if os.path.isfile(p)}
    if not present:
        r.skip("splits", "ImageSets/ is empty — run generate_splits.py to create "
               "train/val/trainval.txt.")
        return

    all_ids = _ids(kitti_dir, "velodyne")

    def read(p: str) -> List[str]:
        with open(p) as fh:
            return [ln.strip() for ln in fh if ln.strip()]

    sets = {n: read(p) for n, p in present.items()}
    # All listed ids must exist on disk.
    for n, ids in sets.items():
        unknown = [i for i in ids if i not in all_ids]
        dup = len(ids) - len(set(ids))
        if unknown:
            r.fail(f"split_{n}_ids", f"{n}.txt references {len(unknown)} id(s) absent on disk "
                   f"(e.g. {unknown[:3]}).")
        elif dup:
            r.warn(f"split_{n}_dupes", f"{n}.txt has {dup} duplicate id(s).")
        else:
            r.ok(f"split_{n}", f"{n}.txt: {len(ids)} ids, all present on disk.",
                 value=float(len(ids)))

    # train/val disjoint and trainval == union.
    if "train" in sets and "val" in sets:
        overlap = set(sets["train"]) & set(sets["val"])
        if overlap:
            r.fail("split_disjoint", f"train and val overlap in {len(overlap)} id(s).")
        else:
            r.ok("split_disjoint", "train and val are disjoint.")
        if "trainval" in sets:
            if set(sets["trainval"]) != set(sets["train"]) | set(sets["val"]):
                r.warn("split_trainval", "trainval.txt is not exactly the union of train and val.")
            else:
                r.ok("split_trainval", "trainval.txt equals train ∪ val.")


def run(kitti_dir: str) -> SectionResult:
    r = SectionResult("structure")
    if not os.path.isdir(kitti_dir):
        r.fail("kitti_dir", f"KITTI directory not found: {kitti_dir}")
        return r
    _check_layout(kitti_dir, r)
    _check_bin_sizes(kitti_dir, r)
    _check_parity_and_contiguity(kitti_dir, r)
    _check_splits(kitti_dir, r)
    return r