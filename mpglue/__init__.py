from .raster_tools import rinfo, mparray

from .vector_tools import vinfo

from .classification import classification, classification_r
from .error_matrix import error_matrix
from .veg_indices import veg_indices, VegIndicesEquations
from .vrt_builder import vrt_builder


__all__ = ['rinfo', 'mparray',
           'vinfo',
           'classification', 'classification_r', 'error_matrix', 'veg_indices', 'VegIndicesEquations',
           'vrt_builder']

__version__ = '0.0.2'
