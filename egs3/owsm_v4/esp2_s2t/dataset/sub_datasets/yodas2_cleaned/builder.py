"""Cleaned YODAS2: the shared cleaned-YODAS build, pointed at yodas2."""

from __future__ import annotations

from egs3.owsm_v4.esp2_s2t.dataset.sub_datasets.yodas_cleaned import (
    YODASCleanedBuilder,
)


class YODAS2CleanedBuilder(YODASCleanedBuilder):
    """Build the cleaned YODAS2 cache from published metadata plus fetched audio."""

    CORPUS = "yodas2_cleaned"
