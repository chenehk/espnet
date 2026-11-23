"""Cleaned YODAS3: the shared cleaned-YODAS build, pointed at yodas3."""

from __future__ import annotations

from egs3.owsm_v4.esp2_s2t.dataset.sub_datasets.yodas_cleaned import (
    YODASCleanedBuilder,
)


class YODAS3CleanedBuilder(YODASCleanedBuilder):
    """Build the cleaned YODAS3 cache from published metadata plus fetched audio."""

    CORPUS = "yodas3_cleaned"
