from .errors import logger, SensorWavelengthError


# Landsat 8
#   coastal blue, blue, gree, red, NIR, SWIR 1, SWIR 2, cirrus
#   skips panchromatic (otherwise 8) and 2 thermal bands
#
# Sentinel2
#   The full, 10 (10m + 20m) or 13 (10m + 20m + 60m) band image.
#   blue, green, red, NIR, --, --, --, red edge, MidIR, FarIR
#   2, 3, 4, 8, 5, 6, 7, 8A, 11, 12, 1, 9, 10
SENSOR_BAND_DICT = {'ASTER': dict(green=1,
                                  red=2,
                                  nir=3,
                                  midir=4,
                                  farir=5),
                    'ASTER-VNIR': dict(green=1,
                                       red=2,
                                       nir=3),
                    'CBERS': dict(blue=1,
                                  green=2,
                                  red=3,
                                  nir=4),
                    'CBERS2': dict(blue=1,
                                   green=2,
                                   red=3,
                                   nir=4),
                    'CitySphere': dict(blue=3,
                                       green=2,
                                       red=1),
                    'GeoEye1': dict(blue=1,
                                    green=2,
                                    red=3,
                                    nir=4),
                    'IKONOS': dict(blue=1,
                                   green=2,
                                   red=3,
                                   nir=4),
                    'Quickbird': dict(blue=1,
                                      green=2,
                                      red=3,
                                      nir=4),
                    'WorldView2': dict(cblue=1,
                                       blue=2,
                                       green=3,
                                       yellow=4,
                                       red=5,
                                       rededge=6,
                                       nir=7,
                                       midir=8),
                    'WorldView3': dict(cblue=1,
                                       blue=2,
                                       green=3,
                                       yellow=4,
                                       red=5,
                                       rededge=6,
                                       nir=7,
                                       midir=8),
                    'Planet': dict(blue=1,
                                   green=2,
                                   red=3,
                                   nir=4),
                    'Landsat-sharp': dict(blue=1,
                                          green=2,
                                          red=3,
                                          nir=4),
                    'Landsat': dict(blue=1,
                                    green=2,
                                    red=3,
                                    nir=4,
                                    midir=5,
                                    farir=6,
                                    pan=8),
                    'Landsat8': dict(cblue=1,
                                     blue=2,
                                     green=3,
                                     red=4,
                                     nir=5,
                                     midir=6,
                                     farir=7,
                                     cirrus=8,
                                     thermal1=9,
                                     thermal2=10,
                                     pan=8),
                    'Landsat-thermal': dict(blue=1,
                                            green=2,
                                            red=3,
                                            nir=4,
                                            midir=5,
                                            farir=7),
                    'MODISc5': dict(blue=3,
                                    green=4,
                                    red=1,
                                    nir=2,
                                    midir=6,
                                    farir=7),
                    'SUPPORTED_VIS': dict(blue=3,
                                          green=4,
                                          red=1,
                                          nir=2,
                                          midir=6,
                                          farir=7),
                    'RapidEye': dict(blue=1,
                                     green=2,
                                     red=3,
                                     rededge=4,
                                     nir=5),
                    'Sentinel2': dict(cblue=1,
                                      blue=2,
                                      green=3,
                                      red=4,
                                      rededge=5,
                                      rededge2=6,
                                      rededge3=7,
                                      niredge=8,
                                      nir=9,
                                      wv=10,
                                      cirrus=11,
                                      midir=12,
                                      farir=13),
                    'Sentinel2-10m': dict(blue=1,
                                          green=2,
                                          red=3,
                                          nir=4),
                    'Sentinel2-20m': dict(rededge=1,
                                          niredge=4,
                                          midir=5,
                                          farir=6),
                    'RGB': dict(blue=3,
                                green=2,
                                red=1),
                    'BGR': dict(blue=1,
                                green=2,
                                red=3),
                    'pan-sharp57': dict(midir=1,
                                        farir=2)}

SUPPORTED_SENSORS = list(SENSOR_BAND_DICT)

# The wavelengths needed to compute the index.
# The wavelengths are loaded in order, so the
#   order should match the equations in
#   ``self.equations``.
VI_WAVELENGTHS = {'ARVI': ['blue', 'red', 'nir'],
                  'BRIGHT': ['green', 'red', 'nir', 'midir'],
                  'CBI': ['cblue', 'blue'],
                  'CIRE': ['rededge', 'rededge3'],
                  'EVI': ['blue', 'red', 'nir'],
                  'EVI2': ['red', 'nir'],
                  'IPVI': ['red', 'nir'],
                  'GNDVI': ['green', 'nir'],
                  'MNDWI': ['midir', 'green'],
                  'MSAVI': ['red', 'nir'],
                  'NDSI': ['nir', 'midir'],
                  'NDBAI': ['midir', 'farir'],
                  'NBRI': ['farir', 'nir'],
                  'NDII': ['ndvi', 'midir', 'farir'],
                  'NDVI': ['red', 'nir'],
                  'RENDVI': ['rededge', 'nir'],
                  'ONDVI': ['red', 'nir'],
                  'NDWI': ['nir', 'green'],
                  'PNDVI': ['green', 'red'],
                  'RBVI': ['blue', 'red'],
                  'GBVI': ['blue', 'green'],
                  'SATVI': ['red', 'midir', 'farir'],
                  'SAVI': ['red', 'nir'],
                  'OSAVI': ['red', 'nir'],
                  'SVI': ['red', 'nir'],
                  'TNDVI': ['red', 'nir'],
                  'TVI': ['green', 'nir'],
                  'TWVI': ['red', 'nir', 'midir'],
                  'YNDVI': ['yellow', 'nir'],
                  'VCI': ['red', 'nir'],
                  'VISMU': ['blue', 'green', 'red'],
                  'WI': ['red', 'midir']}

SUPPORTED_VIS = list(VI_WAVELENGTHS)


def sensor_wavelength_check(sensor2check, wavelengths):

    """
    This function cross-checks wavelengths with a sensor

    Args:
        sensor2check (str): The sensor to check.
        wavelengths (str list): The wavelengths to check.
    """

    for wavelength in wavelengths:

        if wavelength not in SENSOR_BAND_DICT[sensor2check]:

            logger.info('  The sensor is given as {SENSOR}, which does not have wavelength {WV}.'.format(SENSOR=sensor2check,
                                                                                                         WV=wavelength))

            logger.error('  The {WV} is not supported by {SENSOR}.\nPlease specify the correct sensor.'.format(WV=wavelength,
                                                                                                               SENSOR=sensor2check))

            raise SensorWavelengthError


def get_index_bands(spectral_index, sensor):

    """
    Gets the bands needed to process a spectral index.

    Args:
        spectral_index (str): The spectral index to process.
        sensor (str): The satellite sensor.

    Example:
        >>> spectral_bands = get_index_bands('NDVI', 'Landsat')

    Returns:
        A list of bands indices, indexed for GDAL (i.e., 1st position = 1).
    """

    if spectral_index not in SUPPORTED_VIS:

        logger.error('  The spectral index must be one of:  {VIS}'.format(VIS=SUPPORTED_VIS))
        raise NameError

    if sensor not in SUPPORTED_SENSORS:

        logger.error('  The sensor must be one of:  {SENSORS}'.format(SENSORS=SUPPORTED_SENSORS))
        raise NameError

    wavelengths = VI_WAVELENGTHS[spectral_index.upper()]

    sensor_wavelength_check(sensor, wavelengths)

    return [SENSOR_BAND_DICT[sensor][wavelength] for wavelength in wavelengths]
