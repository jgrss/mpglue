# Landsat 8
#   coastal blue, blue, gree, red, NIR, SWIR 1, SWIR 2, cirrus
#   skips panchromatic (otherwise 8) and 2 thermal bands
#
# Sentinel2
#   The full, 10 (10m + 20m) or 13 (10m + 20m + 60m) band image.
#   blue, green, red, NIR, --, --, --, red edge, MidIR, FarIR
#   2, 3, 4, 8, 5, 6, 7, 8A, 11, 12, 1, 9, 10
SENSOR_BAND_DICT = {'ASTER': {'green': 1, 'red': 2, 'nir': 3, 'midir': 4, 'farir': 5},
                    'ASTER-VNIR': {'green': 1, 'red': 2, 'nir': 3},
                    'CBERS': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4},
                    'CBERS2': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4},
                    'CitySphere': {'blue': 3, 'green': 2, 'red': 1},
                    'GeoEye1': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4},
                    'IKONOS': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4},
                    'Quickbird': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4},
                    'WorldView2': {'cblue': 1, 'blue': 2, 'green': 3, 'yellow': 4, 'red': 5,
                                   'rededge': 6, 'nir': 7, 'midir': 8},
                    'WorldView3': {'cblue': 1, 'blue': 2, 'green': 3, 'yellow': 4, 'red': 5,
                                   'rededge': 6, 'nir': 7, 'midir': 8},
                    'Planet': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4},
                    'Landsat-sharp': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4},
                    'Landsat': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4, 'midir': 5, 'farir': 6,
                                'pan': 8},
                    'Landsat8': {'cblue': 1, 'blue': 2, 'green': 3, 'red': 4, 'nir': 5, 'midir': 6,
                                 'farir': 7, 'cirrus': 8, 'thermal1': 9, 'thermal2': 10, 'pan': 8},
                    'Landsat-thermal': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4, 'midir': 5, 'farir': 7},
                    'MODISc5': {'blue': 3, 'green': 4, 'red': 1, 'nir': 2, 'midir': 6, 'farir': 7},
                    'MODIS': {'blue': 10, 'green': 11, 'red': 8, 'nir': 9, 'midir': 13, 'farir': 14},
                    'RapidEye': {'blue': 1, 'green': 2, 'red': 3, 'rededge': 4, 'nir': 5},
                    'Sentinel2': {'cblue': 1, 'blue': 2, 'green': 3, 'red': 4,
                                  'rededge': 5, 'rededge2': 6, 'rededge3': 7, 'niredge': 8,
                                  'nir': 9, 'wv': 10, 'cirrus': 11, 'midir': 12, 'farir': 13},
                    'Sentinel2-10m': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4},
                    'Sentinel2-20m': {'rededge': 1, 'niredge': 4, 'midir': 5, 'farir': 6},
                    'RGB': {'blue': 3, 'green': 2, 'red': 1},
                    'BGR': {'blue': 1, 'green': 2, 'red': 3},
                    'pan-sharp57': {'midir': 1, 'farir': 2}}

SUPPORTED_SENSORS = SENSOR_BAND_DICT.keys()

# The wavelengths needed to compute the index.
# The wavelengths are loaded in order, so the
#   order should match the equations in
#   ``self.equations``.
VI_WAVELENGTHS = {'ARVI': ['blue', 'red', 'nir'],
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
                  'YNDVI': ['yellow', 'nir'],
                  'VCI': ['red', 'nir'],
                  'VISMU': ['blue', 'green', 'red']}

SUPPORTED_VIS = VI_WAVELENGTHS.keys()
