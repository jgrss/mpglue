import logging

_FORMAT = '%(asctime)s:%(levelname)s:%(lineno)s:%(module)s.%(funcName)s:%(message)s'
_formatter = logging.Formatter(_FORMAT, '%H:%M:%S')
_handler = logging.StreamHandler()
_handler.setFormatter(_formatter)

logging.basicConfig(filename='mpglue.log', filemode='w', level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


class LenError(Exception):
    """Raised when lists are not of equal length"""


class ropenError(TypeError):
    """Raised when an object is not an instance of MapPy ropen"""


class ArrayOffsetError(TypeError):
    """Raised when indices are outside the bounds of an array"""
