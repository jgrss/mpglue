from .raster_tools import ropen, read
from .vector_tools import vopen
from .classification.classification import classification, classification_r
from .classification.error_matrix import error_matrix, object_accuracy
from .classification.change import change
from .classification._moving_window import moving_window
from .classification._morph_cells import morph_cells
from .classification.reclassify import reclassify
from .classification.recode import recode
from .classification.sample_raster import sample_raster
from .raster_calc import raster_calc
from .veg_indices import veg_indices, VegIndicesEquations
from .vrt_builder import vrt_builder
from .testing.test import main as test

from .version import __version__

__all__ = ['ropen',
           'read',
           'vopen',
           'classification', 'classification_r',
           'error_matrix', 'object_accuracy',
           'moving_window', 'morph_cells', 'change', 'reclassify', 'recode',
           'raster_calc',
           'veg_indices', 'VegIndicesEquations',
           'vrt_builder',
           'test',
           '__version__']
