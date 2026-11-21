"""VoxPopuli sub-dataset.

``BaseSystem`` looks up ``DatasetBuilder`` and ``Dataset`` by name on the module
named by ``data_src``, which for a dotted reference is this package.
"""

from egs3.owsm_v4.esp2_s2t.dataset.sub_datasets.voxpopuli.builder import (
    VoxPopuliBuilder as DatasetBuilder,
)
from egs3.owsm_v4.esp2_s2t.dataset.sub_datasets.voxpopuli.dataset import (
    VoxPopuliDataset as Dataset,
)

__all__ = ["Dataset", "DatasetBuilder"]
