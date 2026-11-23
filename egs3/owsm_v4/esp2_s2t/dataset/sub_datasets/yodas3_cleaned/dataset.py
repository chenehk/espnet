"""Cleaned YODAS3 read path."""

from __future__ import annotations

from egs3.owsm_v4.esp2_s2t.dataset.sub_datasets.yodas_cleaned import (
    YODASCleanedDataset,
)


class YODAS3CleanedDataset(YODASCleanedDataset):
    """OWSM samples of cleaned YODAS3 built from published metadata."""
