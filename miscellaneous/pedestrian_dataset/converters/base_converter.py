"""
Abstract base class for all Stage-2 format converters.
Each concrete converter (KITTI, nuScenes, Waymo) reads from the Stage-1
intermediate folder and writes to a new output folder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from utils.config_loader import PipelineConfig


class BaseConverter(ABC):

    def __init__(self, config: PipelineConfig, stage1_dir: str, output_dir: str):
        """
        Parameters
        config      : loaded PipelineConfig
        stage1_dir  : path to the Stage-1 intermediate folder for one BAG
        output_dir  : destination folder for this converter's output
        """
        self.config     = config
        self.stage1_dir = stage1_dir
        self.output_dir = output_dir

    @abstractmethod
    def convert(self) -> None:
        """
        Read the Stage-1 data and write the target-format dataset.
        Implementations must be self-contained and must not open BAG files.
        """
        ...

    @property
    def format_name(self) -> str:
        """Human-readable name of the target format (e.g. 'kitti')."""
        return self.__class__.__name__.replace("Converter", "").lower()