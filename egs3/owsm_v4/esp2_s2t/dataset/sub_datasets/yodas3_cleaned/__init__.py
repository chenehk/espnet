"""Cleaned YODAS3 as an OWSM sub-dataset.

``BaseSystem`` looks up ``DatasetBuilder`` and ``Dataset`` by name on the module
named by ``data_src``, which for a dotted reference is this package.
"""

from egs3.owsm_v4.esp2_s2t.dataset.sub_datasets.yodas3_cleaned.builder import (
    YODAS3CleanedBuilder as DatasetBuilder,
)
from egs3.owsm_v4.esp2_s2t.dataset.sub_datasets.yodas3_cleaned.dataset import (
    YODAS3CleanedDataset as Dataset,
)

__all__ = ["Dataset", "DatasetBuilder"]
