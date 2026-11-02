from .base_converter      import BaseConverter
from .kitti_converter     import KITTIConverter
from .nuscenes_converter  import NuScenesConverter
from .waymo_converter     import WaymoConverter

_REGISTRY = {
    "kitti":    KITTIConverter,
    "nuscenes": NuScenesConverter,
    "waymo":    WaymoConverter,
}


def get_converter(fmt: str, config, stage1_dir: str, output_dir: str) -> BaseConverter:
    """
    Return an instantiated converter for "fmt".

    Parameters
    fmt        : one of "kitti", "nuscenes", "waymo"
    config     : PipelineConfig
    stage1_dir : path to the Stage-1 intermediate folder
    output_dir : destination for this converter's output
    """
    cls = _REGISTRY.get(fmt.lower())
    if cls is None:
        raise ValueError(f"Unknown output format: '{fmt}'. Supported: {list(_REGISTRY)}")
    return cls(config, stage1_dir, output_dir)