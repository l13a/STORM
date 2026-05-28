r"""
STORM (Spatial Temporal Omics Regulatory Modeling)

A graph-linked unified embedding model for paired multi-omics spatial data,
derived from `GLUE`_ (Cao & Gao, 2022) with gene-program masking adapted
from `NicheCompass`_ and a temporal-alignment objective for time-resolved
samples.

.. _GLUE: https://github.com/gao-lab/GLUE
.. _NicheCompass: https://github.com/Lotfollahi-lab/nichecompass
"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ModuleNotFoundError:
    from pkg_resources import get_distribution, DistributionNotFound as PackageNotFoundError
    version = lambda name: get_distribution(name).version

from . import (
    clustering,
    data,
    evaluation,
    genomics,
    graph,
    models,
    num,
    plot,
    preprocessing,
    programs,
)
from ._internal import config, log


name = "storm"
try:
    __version__ = version(name)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
