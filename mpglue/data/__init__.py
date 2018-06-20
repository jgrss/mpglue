import os

from ..paths import get_main_path


mpglue_path = get_main_path()

landsat_gtiff = os.path.join(mpglue_path, 'data', '225r85_etm_2000_0424.tif')
landsat_vrt = os.path.join(mpglue_path, 'data', '225r85_etm_2000_0424.vrt')
