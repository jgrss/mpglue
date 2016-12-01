class LenError(Exception):
    """Raised when lists are not of equal length"""


class RinfoError(TypeError):
    """Raised when an object is not an instance of MapPy rinfo"""


class ArrayOffsetError(TypeError):
    """Raised when indices are outside the bounds of an array"""
