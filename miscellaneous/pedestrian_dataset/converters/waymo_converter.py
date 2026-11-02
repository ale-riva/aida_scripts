"""
Stage-2 converter: intermediate format -> Waymo Open Dataset TFRecord format.
"""

from __future__ import annotations

import logging
import os

from .base_converter import BaseConverter

logger = logging.getLogger(__name__)


class WaymoConverter(BaseConverter):

    def convert(self) -> None:
        logger.info("[Waymo] Starting conversion from: %s", self.stage1_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        # TODO (Step 7)
        raise NotImplementedError("WaymoConverter.convert() - to be implemented.")