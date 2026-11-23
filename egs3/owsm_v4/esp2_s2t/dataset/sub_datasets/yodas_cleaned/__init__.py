"""Shared implementation for the cleaned-YODAS corpora.

Not a sub-dataset: it has no ``config.yaml`` and exports no ``DatasetBuilder``
or ``Dataset``, so it cannot be named as a ``data_src``. The corpora built on
it are ``yodas2_cleaned`` and ``yodas3_cleaned``, which differ only in the
metadata repository they read and the audio repository that implies. Keeping
the source on the class rather than in a module constant is what lets both
appear in one mixture.
"""

from egs3.owsm_v4.esp2_s2t.dataset.sub_datasets.yodas_cleaned.builder import (
    YODASCleanedBuilder,
)
from egs3.owsm_v4.esp2_s2t.dataset.sub_datasets.yodas_cleaned.dataset import (
    YODASCleanedDataset,
)

__all__ = ["YODASCleanedBuilder", "YODASCleanedDataset"]
