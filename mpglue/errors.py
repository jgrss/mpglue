import os
import logging

from .paths import get_main_path

_FORMAT = '%(asctime)s:%(levelname)s:%(lineno)s:%(module)s.%(funcName)s:%(message)s'
_formatter = logging.Formatter(_FORMAT, '%H:%M:%S')
_handler = logging.StreamHandler()
_handler.setFormatter(_formatter)

logging.basicConfig(filename=os.path.join(get_main_path(), 'mpglue.log'),
                    filemode='w',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


class ArrayOffsetError(OverflowError):
    """Raised when indices are outside the bounds of an array"""


class EmptyImage(OSError):
    """Raised when an input image is empty."""


class LenError(AssertionError):
    """Raised when lists are not of equal length"""


class MissingRequirement(KeyError):
    """Raised when a required parameter is missing"""


class ropenError(TypeError):
    """Raised when an object is not an instance of MapPy ropen"""


class TransformError(ValueError):
    """Raised when coordinates cannot be transformed via ogr"""


class ArrayShapeError(AssertionError):
    """Raised when an array does not conform to correct shape requirements"""


class SensorWavelengthError(NameError):
    """Raised when a sensor does not support a wavelength"""
