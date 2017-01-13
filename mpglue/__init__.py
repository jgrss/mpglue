from .raster_tools import ropen, read
from .vector_tools import vopen

from .classification.classification import classification, classification_r
from .classification.error_matrix import error_matrix
from .classification.sample_raster import sample_raster

from .veg_indices import veg_indices, VegIndicesEquations
from .vrt_builder import vrt_builder


__all__ = ['ropen', 'read',
           'vopen',
           'classification', 'classification_r', 'error_matrix', 'veg_indices', 'VegIndicesEquations',
           'vrt_builder']

__version__ = '0.0.6'
