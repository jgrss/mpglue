#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 9/24/2011
"""

from __future__ import division, print_function
from future.utils import viewitems
from builtins import int, map

import math
from copy import copy
import datetime
from collections import OrderedDict
import calendar

# MapPy
from . import raster_tools

# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')

# Pandas
try:
    import pandas as pd
except ImportError:
    raise ImportError('Pandas must be installed')

# Numexpr
try:
    import numexpr as ne
except ImportError:
    raise ImportError('Numexpr must be installed')

# Scikit-image
try:
    from skimage.exposure import rescale_intensity
except ImportError:
    raise ImportError('Scikit-image must be installed')

old_settings = np.seterr(all='ignore')


def earth_sun_distance(julian_day):

    """
    Converts Julian Day to earth-sun distance.

    Args:
        julian_day (int): The Julian Day.
        
    Returns:
        Earth-Sun distance (d) in astronomical units for Day of the Year (DOY)
    """

    return (1.0 - 0.01672 * math.cos(math.radians(0.9856 * (float(julian_day) - 4.0)))) ** 2.0


def julian_day_dictionary(start_year=1980, end_year=2050, store='st_jd'):

    """
    A function to setup standard (continuously increasing) Julian Days

    Args:
        start_year (Optional[int])
        end_year (Optional[int])
        store (Optional[str]): Choices are ['st_jd', 'date'].

    Returns:
        Dictionary of {'yyyy-doy': yyyydoy}

        or

        Dictionary of {'yyyy-doy': 'yyyy.mm.dd'}
    """

    jd_dict = OrderedDict()

    for yyyy in range(start_year, end_year):

        # Get the days for the current year.
        time_stamp = pd.date_range('{:d}-01-01'.format(yyyy),
                                   '{:d}-12-31'.format(yyyy),
                                   name='time',
                                   freq='D')

        date_time = time_stamp.to_pydatetime()

        dd_yyyy = ('-{:d},'.format(yyyy).join(map(str, [dt.timetuple().tm_yday
                                                        for dt in date_time])) + '-{:d}'.format(yyyy)).split(',')

        if store == 'date':
            
            date_list = list(map(str, ['{}.{:02d}.{:03d}.{:03d}'.format(dt.timetuple().tm_year,
                                                                        int(dt.timetuple().tm_mon),
                                                                        int(dt.timetuple().tm_mday),
                                                                        int(dt.timetuple().tm_yday)) for dt in
                                       date_time]))

        for date in range(0, len(dd_yyyy)):

            if store == 'st_jd':

                date_split = dd_yyyy[date].split('-')

                dd_ = '{:03d}'.format(int(date_split[0]))
                yyyy_ = '{}'.format(date_split[1])

                jd_dict['{}-{}'.format(yyyy_, dd_)] = int('{}{}'.format(yyyy_, dd_))

            elif store == 'date':

                y, m, d, jd = date_list[date].split('.')

                jd_dict['{}-{}'.format(y, jd)] = '{}.{}.{}'.format(y, m, d)

    return jd_dict


def julian_day_dictionary_r(start_year=1980, end_year=2050, jd_dict=None):

    """
    A function to get the reverse Julian Data dictionary

    Args:
        start_year (Optional[int])
        end_year (Optional[int])
        jd_dict (Optional[dict]): A pre-calculated (i.e., from `julian_day_dictionary`) Julian day dictionary.

    Returns:
        Dictionary of {yyyyddd: 'year-day'}
    """

    jd_dict_r = OrderedDict()

    if not isinstance(jd_dict, dict):

        jd_dict = julian_day_dictionary(start_year=start_year,
                                        end_year=end_year)

    for k, v in viewitems(jd_dict):
        jd_dict_r[v] = k

    return jd_dict_r


def get_leap_years(start_year=1980, end_year=2050):

    """
    Gets the number of calendar days by year, either 365 or 366

    Args:
        start_year (Optional[int])
        end_year (Optional[int])

    Returns:
        Dictionary, with keys --> values as yyyy --> n days
    """

    leap_year_dict = dict()

    for yyyy in range(start_year, end_year):

        if calendar.isleap(yyyy):
            leap_year_dict[yyyy] = 366
        else:
            leap_year_dict[yyyy] = 365

    return leap_year_dict


def jd_interp(the_array, length, skip_factor):

    """
    Args:
        the_array
        length
        skip_factor     
    """

    # Get a dictionary of leap years.
    year_dict = get_leap_years()

    # The first year.
    current_year = int(str(the_array[0])[:4])

    # The Julian day `yyyyddd`.
    the_date = the_array[0]

    rescaled = list()

    for i in range(0, length):

        # Append `yyyyddd`.
        rescaled.append(the_date)

        the_date += skip_factor

        current_doy = int(str(the_date)[4:])

        # Check year overload.
        #   Update calendar day > maximum
        #   days in year `current_year`.
        if current_doy > year_dict[current_year]:

            current_doy_diff = current_doy - year_dict[current_year]

            current_year += 1

            the_date = int('{:d}{:03d}'.format(current_year, current_doy_diff))

    return rescaled


def rescale_scaled_jds(the_array, counter=1000):

    """
    Rescales `yyyyddd` values to monotonically increasing values.
            
    Args:
        the_array (1d array-like, str or int): The Julian day list.
        counter (Optional[int]): The starting index.
    """

    iter_length = len(the_array) - 1

    # Get a dictionary of leap years.
    year_dict = get_leap_years()

    # Get the first year.
    current_year = int(str(the_array[0])[:4])

    rescaled = list()

    for i in range(0, iter_length):

        rescaled.append(counter)

        next_year = int(str(the_array[i+1])[:4])

        if next_year != current_year:

            # Next year Julian Day + (Current year max - current year Julian Day)
            counter += int(str(the_array[i+1])[4:]) + (year_dict[current_year] - int(str(the_array[i])[4:]))

        else:

            # Next Julian Day - Current Julian Day
            counter += the_array[i+1] - the_array[i]

        current_year = copy(next_year)

    rescaled.append(counter)

    return rescaled


def date2julian(month, day, year):

    """
    Converts month, day, and year to Julian Day.

    Args:
        month (int or str): The month.
        day (int or str): The day.
        year (int or str): The year.

    Returns:
        Julian Day
    """

    # Convert strings to integers
    month = int(month)
    day = int(day)
    year = int(year)

    fmt = '%Y.%m.%d'

    dt = datetime.datetime.strptime('{}.{}.{}'.format(str(year), str(month), str(day)), fmt)

    return int(dt.timetuple().tm_yday)


def julian2date(julian_day, year, jd_dict_date=None):

    """
    Converts Julian day to month and day.

    Args:
        julian_day (int or str): The Julian Day.
        year (int or str): The year.
        jd_dict_date (Optional[dict]): A pre-calculated (i.e., from `julian_day_dictionary`) Julian day dictionary.

    Returns:
        (month, day) of the Julian Day `julian_day`.
    """

    year = int(year)
    julian_day = int(julian_day)

    if not isinstance(jd_dict_date, dict):
        jd_dict_date = julian_day_dictionary(store='date')

    y, m, d = jd_dict_date['{:d}-{:03d}'.format(year, julian_day)].split('.')

    return int(m), int(d)


def yyyyddd2months(yyyyddd_list):

    """
    Converts yyyyddd format to yyyymmm format.
    """

    jd_dict_date = julian_day_dictionary(store='date')

    return ['{}{:03d}'.format(str(yyyyddd)[:4],
                              julian2date(str(yyyyddd)[4:],
                                          str(yyyyddd)[:4],
                                          jd_dict_date=jd_dict_date)[0]) for yyyyddd in yyyyddd_list]


def scaled_jd2jd(scaled_jds, return_jd=True):

    """
    Converts scaled Julian day integers to string yyyy-ddd format.

    Args:
        scaled_jds (int list): The Julian days to convert.
        return_jd (Optional[bool]): Whether to return Julian Days as 'yyyy-ddd'.
            Otherwise, returns month-day-year format. Default is True.
    """

    jd_dict_r = julian_day_dictionary_r()

    xd_smooth_labels = list()

    for k in scaled_jds:

        if int(k) in jd_dict_r:
            xd_smooth_labels.append(jd_dict_r[int(k)])
        else:

            # Check to see if the day of year is 366,
            #   but the year is not a leap year.

            yyyy = int(str(k)[:4])
            doy = int(str(k)[4:])

            if doy == 366:

                new_k = int('{:d}001'.format(yyyy+1))
                xd_smooth_labels.append(jd_dict_r[new_k])

    if return_jd:
        return xd_smooth_labels
    else:

        jd_dict_date = julian_day_dictionary(store='date')

        return ['{}-{}-{}'.format(julian2date(l.split('-')[1], l.split('-')[0], jd_dict_date=jd_dict_date)[0],
                                  julian2date(l.split('-')[1], l.split('-')[0], jd_dict_date=jd_dict_date)[1],
                                  l.split('-')[0]) for l in xd_smooth_labels]


class Conversions(object):

    """
    A class for sensor-specific radiometric calibration
    """

    def dn2radiance(self, dn_array, band, aster_gain_setting='high',
                    cbers_series='CBERS2B', cbers_sensor='HRCCD', landsat_gain=None, landsat_bias=None,
                    wv2_abs_calibration_factor=None, wv2_effective_bandwidth=None):

        """
        Converts digital numbers (DN) to radiance

        Args:
            dn_array (ndarray): The array to calibrate.
            band (int): The band to calibrate.
            aster_gain_setting (Optional[str]): The gain setting for ASTER. Default is 'high'.
            cbers_series (Optional[str]): The CBERS series. Default is 'CBERS2B'.
            cbers_sensor (Optional[str]): The CBERS sensor. Default is 'HRCCD'.
            landsat_gain (Optional[float]): The gain setting for Landsat. Default is None.
            landsat_bias (Optional[float]): The bias setting for Landsat. Default is None.
            wv2_abs_calibration_factor (Optional[float]): The absolute calibration factor for WorldView2. Default is None.
            wv2_effective_bandwidth (Optional[float]): The effective bandwidth for WorldView2. Default is None.

        Formulas:

            ASTER:
                L = (DN-1) * UCC

                Where,
                    L   = Spectral radiance at the sensor's aperture
                    UCC = Unit Conversion Coefficient

            CBERS:
                L = DN / CC

                Where,
                    L   = Spectral radiance at the sensor's aperture
                    DN  = Quantized calibrated pixel value
                    CC  = Absolute calibration coefficient

            WorldView2:
                L = (absCalFactor * DN) / eB

                Where,
                    L = TOA spectral radiance
                    absCalFactor = Absolute radiometric calibration factor
                    DN = digital counts
                    eB = Effective bandwidth

        Returns:
            Radiance as ndarray.
        """

        if self.sensor == 'ASTER':

            gain_setting_dict = {'high': 0, 'normal': 1, 'low1': 2, 'low2': 3}

            ucc = np.array([[.676, 1.688, 2.25, .0],
                            [.708, 1.415, 1.89, .0],
                            [.423, .862, 1.15, .0],
                            [.1087, .2174, .2900, .2900],
                            [.0348, .0696, .0925, .4090],
                            [.0313, .0625, .0830, .3900],
                            [.0299, .0597, .0795, .3320],
                            [.0209, .0417, .0556, .2450], 
                            [.0159, .0318, .0424, .2650]], dtype='float32')

            radiance = np.float32(np.subtract(dn_array, 1.))
            radiance = np.float32(np.multiply(radiance, ucc[int(band)-1][gain_setting_dict[aster_gain_setting]]))

        elif self.sensor == 'CBERS':

            if '2B' in cbers_series and 'HRCCD' in cbers_sensor:

                ucc = {'1': .97, '2': 1.74, '3': 1.083, '4': 2.105}

            else:
                raise NameError('\nSeries not recoginized.\n')

            radiance = np.float32(np.divide(dn_array, ucc[str(band)]))

        elif self.sensor.lower() in ['tm', 'etm', 'oli_tirs']:

            if not isinstance(landsat_gain, float) or not isinstance(landsat_bias, float):
                raise ValueError('\nCalibration coefficients not set.\n')

            radiance = ne.evaluate('(landsat_gain * dn_array) + landsat_bias')

        elif self.sensor == 'WorldView2':

            if not wv2_abs_calibration_factor or not wv2_effective_bandwidth:
                raise ValueError('\nCalibration coefficients not set.\n')

            radiance = np.float32(np.divide(np.multiply(wv2_abs_calibration_factor[band-1], dn_array),
                                            wv2_effective_bandwidth[band-1]))

        radiance[dn_array <= 0] = 0

        return radiance

    def radiance2reflectance(self, radiance_array, band, solar_angle=None, julian_day=None, bd_esun=None,
                             landsat_gain=None, landsat_bias=None, aster_solar_scheme='Smith',
                             cbers_series='CBERS2', cbers_sensor='HRCCD'):

        """
        Converts radiance to top of atmosphere reflectance

        Args:
            radiance_array (ndarray): The ndarray to calibrate.
            band (Optional[int]): The band to calibrate. Default is 1.
            solar_angle (float)
            julian_day (int)
            bd_esun (float):
            landsat_gain (float)
            landsat_bias (float)
            aster_solar_scheme (Optional[str]): The solar scheme for ASTER. Default is 'Smith'.
                Choices are ['Smith', 'ThomeEtAlA', 'TomeEtAlB'].
            cbers_series (Optional[str]): The CBERS series. Default is 'CBERS2B'.
            cbers_sensor (Optional[str]): The CBERS sensor. Default is 'HRCCD'.

        Returns:
            Reflectance as ndarray.
        """

        d_sq = earth_sun_distance(julian_day)
        pi = math.pi
        cos0 = np.cos(np.radians(90. - solar_angle))

        if self.sensor == 'ASTER':

            scheme = {'Smith': 0,
                      'ThomeEtAlA': 1,
                      'ThomeEtAlB': 2}

            # Solar spectral irradiance values
            #   for each ASTER band.
            esun = np.array([[1845.99, 1847., 1848.],
                             [1555.74, 1553., 1549.],
                             [1119.47, 1118., 1114.],
                             [231.25, 232.5,  225.4 ],
                             [79.81,  80.32,  86.63 ],
                             [74.99,  74.92,  81.85 ],
                             [68.66,  69.20,  74.85 ],
                             [59.74,  59.82,  66.49 ],
                             [56.92,  57.32,  59.85]], dtype='float32')

            bd_esun = esun[band-1][scheme[aster_solar_scheme]]

        elif self.sensor == 'CBERS2':

            # Solar spectral irradiance values for each sensor
            if '2B' in cbers_series and 'HRCCD' in cbers_sensor:

                esun = {1: 1934.03, 2: 1787.1, 3: 1548.97, 4: 1069.21}

                bd_esun = esun[band]

        elif self.sensor == 'WorldView2':

            # band names
            # coastal, blue, green, yellow, red, red edge, NIR1, NIR2
            esun = {'BAND_P': 1580.814, 'BAND_C': 1758.2229, 'BAND_B': 1974.2416, 'BAND_G': 1856.4104,
                    'BAND_Y': 1738.4791, 'BAND_R': 1559.4555, 'BAND_RE': 1342.0695, 'BAND_N': 1069.7302,
                    'BAND_N2': 861.2866}

            band_positions = {1: 'BAND_P', 2: 'BAND_C', 3: 'BAND_B', 4: 'BAND_G', 5: 'BAND_Y', 6: 'BAND_R',
                              7: 'BAND_RE', 8: 'BAND_N', 9: 'BAND_N2'}

            bd_esun = esun[band_positions[band]]

        else:
            raise NameError('\n{} is not a supported sensor.'.format(self.sensor))

        if self.sensor.lower() == 'oli_tirs':
            reflectance_equation = '((radiance_array * landsat_gain) + landsat_bias) / cos0'
        else:
            reflectance_equation = '(radiance_array * pi * d_sq) / (bd_esun * cos0)'

        reflectance = ne.evaluate(reflectance_equation)

        reflectance[radiance_array <= 0] = 0.

        return reflectance

    def get_gain_bias(self, series, sensor, band_position, l_max, l_min, coeff_check):

        max_min_dict = {'TM4': {'lmax': {'1': 163., '2': 336., '3': 254., '4': 221.,
                                         '5': 31.4, '6': 15.3032, '7': 16.6},
                                'lmin': {'1': -1.52, '2': -2.84, '3': -1.17, '4': -1.51,
                                         '5': -.37, '6': 1.2378, '7': -.15}},
                        'TM5': {'lmax': {'1': 169., '2': 333., '3': 264., '4': 221.,
                                         '5': 30.2, '6': 15.3032, '7': 16.5},
                                'lmin': {'1': -1.52, '2': -2.84, '3': -1.17, '4': -1.51,
                                         '5': -.37, '6': 1.2378, '7': -.15}},
                        'ETM': {'lmax': {'1': 191.6, '2': 196.6, '3': 152.9, '4': 157.4,
                                         '5': 31.06, '6': 12.65, '7': 10.8},
                                'lmin': {'1': -6.2, '2': -6.4, '3': -5., '4': -5.1,
                                         '5': -1., '6': 3.2, '7': -.35}}}

        # Get standard coefficients if none were
        #   obtained from the metadata.
        if coeff_check == 999:

            if 'sat4' in series or 'LANDSAT_4' in series and 'TM' in sensor:
                sensor_series = 'TM4'
            elif 'sat5' in series or 'LANDSAT_5' in series and 'TM' in sensor:
                sensor_series = 'TM5'
            elif 'sat7' in series or 'LANDSAT_7' in series:
                sensor_series = 'ETM'
            else:
                raise NameError('The Landsat sensor could not be found.')

            l_max = max_min_dict[sensor_series]['lmax'][str(band_position)]
            l_min = max_min_dict[sensor_series]['lmin'][str(band_position)]

        gain = (l_max - l_min) / 254.
        bias = l_min - ((l_max - l_min) / 254.)

        return gain, bias

    def get_kelvin_coefficients(self, series, sensor, band_position):

        k_dict = {'TM4': {'k1': 671.62, 'k2': 1284.3},
                  'TM5': {'k1': 607.76, 'k2': 1260.56},
                  'ETM': {'k1': 666.09, 'k2': 1282.71}}

        if 'sat4' in series or 'LANDSAT_4' in series and 'TM' in sensor:
            sensor_series = 'TM4'
        elif 'sat5' in series or 'LANDSAT_5' in series and 'TM' in sensor:
            sensor_series = 'TM5'
        elif 'sat7' in series or 'LANDSAT_7' in series:
            sensor_series = 'ETM'
        else:
            raise NameError('The Landsat sensor could not be found.')

        k1 = k_dict[sensor_series]['k1']
        k2 = k_dict[sensor_series]['k2']

        return k1, k2

    def get_esun(self, series, sensor, band_position):

        esun_dict = {'TM4': {'1': 1983., '2': 1795., '3': 1539., '4': 1028., '5': 219.8, '7': 83.49},
                     'TM5': {'1': 1983., '2': 1796., '3': 1536., '4': 1031., '5': 220., '7': 83.44},
                     'ETM': {'1': 1997., '2': 1812., '3': 1533., '4': 1039., '5': 230.8, '7': 84.9}}

        # Solar spectral irradiance values for Landsat sensors.
        if 'sat4' in series or 'LANDSAT_4' in series and 'TM' in sensor:
            sensor_series = 'TM4'
        elif 'sat5' in series or 'LANDSAT_5' in series and 'TM' in sensor:
            sensor_series = 'TM5'
        elif 'sat7' in series or 'LANDSAT_7' in series:
            sensor_series = 'ETM'
        else:
            raise NameError('The Landsat sensor could not be found.')

        return esun_dict[sensor_series][str(band_position)]

    def get_dn_dark(self, dn_array, min_dark):

        val = 1
        min_true = True

        while min_true:

            idx = np.where(dn_array == val)

            if len(dn_array[idx]) >= min_dark:
                min_true = False
            else:
                val += 1

        return val

    def get_tri(self, series, sensor, band_position):

        tri_dict = {'TM4': {'1': .485, '2': .569, '3': .659, '4': .841, '5': 1.676, '7': 2.222},
                    'TM5': {'1': .485, '2': .569, '3': .666, '4': .84, '5': 1.676, '7': 2.223},
                    'ETM': {'1': .483, '2': .56, '3': .662, '4': .835, '5': 1.648, '7': 2.206}}

        if 'sat4' in series or 'LANDSAT_4' in series and 'TM' in sensor:
            sensor_series = 'TM4'
        elif 'sat5' in series or 'LANDSAT_5' in series and 'TM' in sensor:
            sensor_series = 'TM5'
        elif 'sat7' in series or 'LANDSAT_7' in series:
            sensor_series = 'ETM'
        else:
            raise NameError('The Landsat sensor could not be found.')

        return tri_dict[sensor_series][str(band_position)]

    def get_tr(self, tri):

        return .008569 * pow(tri, -4) * (1. + .0113 * pow(tri, -2) + .00013 * pow(tri, -4))

    def get_path_rad(self, gain, bias, dn_dark, bd_esun, cos0, tz, tv, edown, d_sq):

        return (gain * dn_dark) + bias - .01 * (bd_esun * cos0 * tz + edown) * tv / math.pi

    def prepare_dark(self, dn_array, band_position, bd_esun, gain, bias, sensor_angle, dn_dark, min_dark):

        print('\nGetting dark haze value for {} ...\n'.format(self.calibration))

        if dn_dark == -999:
            dn_dark = self.get_dn_dark(dn_array, min_dark)

        print('  Band {:d} haze value: {:d}\n'.format(band_position, dn_dark))

        self.d_sq = earth_sun_distance(self.julian_day)

        # Cosine of solar zenith angle.
        self.cos0 = np.cos(np.radians(90. - self.solar_angle))

        # Cosine of sensor viewing angle (90 degrees
        #   for nadir viewing sensor).
        self.cosS = np.cos(np.radians(90. - sensor_angle))

        tr = self.get_tr(self.get_tri(self.pr.series, raster_tools.SENSOR_DICT[self.pr.sensor.lower()], band_position))

        self.tv = math.exp(-tr / self.cosS)
        self.tz = math.exp(-tr / self.cos0)
        self.edown = .01

        self.path_radiance = self.get_path_rad(gain, bias, dn_dark, bd_esun, self.cos0,
                                               self.tz, self.tv, self.edown, self.d_sq)

    def radiance2reflectance_dos(self, radiance_array, band_position, bd_esun, gain, bias,
                                 sensor_angle=90., dn_dark=-999, min_dark=1000):

        self.prepare_dark(radiance_array, band_position, bd_esun, gain, bias, sensor_angle, dn_dark, min_dark)

        pi = math.pi
        d_sq = self.d_sq
        path_radiance = self.path_radiance
        tv = self.tv
        tz = self.tz
        cos0 = self.cos0
        edown = self.edown

        reflectance_equation = '(radiance_array * pi * d_sq) / (bd_esun * cos0)'

        dos_equation = '(pi * (reflectance - path_radiance)) / (tv * (bd_esun * cos0 * tz * edown))'

        reflectance = ne.evaluate(reflectance_equation)
        reflectance = ne.evaluate(dos_equation)

        reflectance[radiance_array <= 0] = 0.

        return reflectance

    def radiance2kelvin(self, dn_array, k1=None, k2=None):

        temperature = ne.evaluate('k2 / log((k1 / dn_array) + 1)')

        temperature[dn_array <= 0] = 0.

        return temperature


class CalibrateSensor(Conversions):

    """
    A class for radiometric calibration

    Args:
        input_image (str): An image (single or multi-band) to process.
        sensor (Optional[str]): The sensor to calibrate.
            Choices are ['TM', 'ETM', 'OLI_TIRS', 'ASTER', 'CBERS', 'WorldView2'].
        image_date (str): yyyy/mm/dd
        solar_angle (float)
        bands2process (Optional[int or int list]): Default is -1, or all bands.

    Examples:
        >>> from mappy import rad_calibration
        >>>
        >>> # Convert ASTER to radiance.
        >>> cal = rad_calibration.CalibrateSensor('/in_image.tif', 'ASTER')
        >>> cal.process('/out_image.tif', calibration='radiance')
        >>>
        >>> # Convert Landsat to top of atmosphere reflectance.
        >>> cal = rad_calibration.CalibrateSensor('/in_image.tif', 'TM')
        >>> cal.process('/out_image.tif', calibration='toar', metadata='/metadata.MTL')
    """

    def __init__(self, input_image, sensor, bands2process=-1):

        self.input_image = input_image
        self.sensor = sensor
        self.bands2process = bands2process

        self.i_info = raster_tools.ropen(self.input_image)

        Conversions.__init__(self)

    def process(self, output_image, image_date=None, solar_angle=None, calibration='radiance', d_type='float32',
                bd_esun_list=[], aster_gain_setting='high', aster_solar_scheme='Smith', cbers_series='CBERS2B',
                cbers_sensor='HRCCD', landsat_gain_list=[], landsat_bias_list=[], k1=None, k2=None,
                wv2_abs_calibration_factor=[], wv2_effective_bandwidth=[], metadata=None):

        """
        Args:
            output_image (str): The output image.
            calibration (Optional[str]): Choices are ['radiance', 'toar', 'cost', 'dos2', 'dos3', 'dos4', 'temp'].
            d_type (Optional[str]): The output storage type. Default is 'float32'.
                Choices are ['float32', 'byte', 'uint16'].
            bd_esun_list (Optional[float list]): A list of ESUN coefficients (for each band to process)
                if ``metadata`` is not given.
            aster_gain_setting (Optional[str]):
            aster_solar_scheme (Optional[str]):
            cbers_series (Optional[str]):
            cbers_sensor (Optional[str]):
            landsat_gain_list (Optional[float list]): A list of gain coefficients (for each band to process)
                if ``metadata`` is not given.
            landsat_bias_list (Optional[float list]): Same as above, with bias.
            wv2_abs_calibration_factor (Optional[str]):
            wv2_effective_bandwidth (Optional[str]):
            metadata (Optional[object or str): A metadata file or object instance. Default is None.

        References:
            Chavez (1988)
            Schroeder et al. (2006), RSE
            Song et al. (2001)
            GRASS manual (http://grass.osgeo.org/grass65/manuals/html65_user/i.landsat.toar.html)

            p = pi(Lsat - Lp) / Tv(Eo * cos(0) * Tz + Edown)

            where,
                p       = At-sensor reflectance

                Lsat    = At-sensor Radiance
                Lp      = Path Radiance
                            = G * DNdark + B-.01(Eo * cos(0) * Tz + Edown)Tv/pi
                Tv      = Atmospheric transmittance from the target toward the sensor
                            = exp(-pi / cos(satellite zenith angle))
                Tz      = Atmospheric transmittance in the illumination direction
                            = exp(-pi / cos(solar zenith angle))
                Eo      = Exoatmospheric solar constant
                0       = Zolar zenith angle
                Edown   = Downwelling diffuse irradiance
        """

        self.output_image = output_image
        self.calibration = calibration
        self.image_date = image_date
        self.solar_angle = solar_angle
        self.d_type = d_type
        self.metadata = metadata
        self.bd_esun_list = bd_esun_list
        self.landsat_gain_list = landsat_gain_list
        self.landsat_bias_list = landsat_bias_list
        self.k1 = k1
        self.k2 = k2

        self.landsat_sensors = ['tm', 'etm', 'oli_tirs']

        # Search for metadata.
        self.get_metadata()

        self.rad_settings = dict(aster_gain_setting=aster_gain_setting,
                                 cbers_series=cbers_series, cbers_sensor=cbers_sensor,
                                 landsat_gain=None, landsat_bias=None,
                                 wv2_abs_calibration_factor=wv2_abs_calibration_factor,
                                 wv2_effective_bandwidth=wv2_effective_bandwidth)

        self.refl_settings = dict(solar_angle=self.solar_angle, julian_day=self.julian_day,
                                  bd_esun=None, landsat_gain=None, landsat_bias=None,
                                  aster_solar_scheme=aster_solar_scheme,
                                  cbers_series=cbers_series, cbers_sensor=cbers_sensor)

        self.temp_settings = dict(k1=self.k1, k2=self.k2)

        self.create_output()

        row_block_size, col_block_size = raster_tools.block_dimensions(self.i_info.rows, self.i_info.cols)

        for i in range(0, self.i_info.rows, row_block_size):

            n_rows = raster_tools.n_rows_cols(i, row_block_size, self.i_info.rows)

            for j in range(0, self.i_info.cols, col_block_size):

                n_cols = raster_tools.n_rows_cols(j, col_block_size, self.i_info.cols)

                # Read the array block.
                dn_array = self.i_info.read(bands2open=self.bands2process,
                                               i=i, j=j,
                                               rows=n_rows, cols=n_cols,
                                               d_type='float32')

                for out_band, band_position, dn_array in zip(self.band_range, self.bands2process, dn_array):

                    # Update radiance settings.
                    self.update_rad_settings(band_position)

                    # Convert DN to radiance.
                    if self.sensor.lower() != 'oli_tirs':
                        cal_array = self.dn2radiance(dn_array, band_position, **self.rad_settings)
                    elif (self.sensor.lower() == 'oli_tirs') and (self.calibration.lower() == 'radiance'):
                        cal_array = self.dn2radiance(dn_array, band_position, **self.rad_settings)

                    if self.calibration.lower() != 'radiance':
                        self.update_toar_settings(band_position)

                    if self.calibration.lower() == 'toar':

                        # Convert radiance to top of atmosphere reflectance.
                        if self.sensor.lower() == 'oli_tirs':
                            cal_array = self.radiance2reflectance(dn_array, band_position, **self.refl_settings)
                        else:
                            cal_array = self.radiance2reflectance(cal_array, band_position, **self.refl_settings)

                    elif self.calibration.lower() == 'dos':

                        cal_array = self.radiance2reflectance_dos(cal_array, band_position,
                                                                  self.refl_settings['bd_esun'],
                                                                  self.refl_settings['landsat_gain'],
                                                                  self.refl_settings['landsat_bias'],
                                                                  sensor_angle=90.,
                                                                  dn_dark=-999, min_dark=1000)

                    elif self.calibration.lower() == 'temperature':

                        cal_array = self.radiance2kelvin(dn_array, **self.temp_settings)

                    # Scale the data to byte or uint16 storage.
                    if self.d_type != 'float32':
                        cal_array = self.scale_data(cal_array)

                    self.out_rst.write_array(cal_array, i=i, j=j, band=out_band)

                    self.out_rst.close_band()

        # Close the input image.
        self.i_info.close()

        # Close the output drivers.
        self.out_rst.close_file()

        self.out_rst = None

    def update_toar_settings(self, band_position):

        # Landsat settings
        if self.sensor.lower() in self.landsat_sensors:

            # Is there a user provided file or object?
            if isinstance(self.metadata, str) or isinstance(self.metadata, raster_tools.LandsatParser):

                if raster_tools.SENSOR_DICT[self.pr.sensor.lower()] == 'oli_tirs':

                    k1 = self.pr.k1
                    k2 = self.pr.k2

                else:

                    self.refl_settings['bd_esun'] = self.get_esun(self.pr.series,
                                                                  raster_tools.SENSOR_DICT[self.pr.sensor.lower()],
                                                                  band_position)

                    k1, k2 = self.get_kelvin_coefficients(self.pr.series,
                                                          raster_tools.SENSOR_DICT[self.pr.sensor.lower()],
                                                          band_position)

                self.temp_settings['k1'] = k1
                self.temp_settings['k2'] = k2

            else:

                bi = self.bands2process[self.bands2process.index(band_position)]

                self.refl_settings['bd_esun'] = self.bd_esun_list[bi]

                self.temp_settings['k1'] = self.k1
                self.temp_settings['k2'] = self.k2

            self.refl_settings['landsat_gain'] = self.rad_settings['landsat_gain']
            self.refl_settings['landsat_bias'] = self.rad_settings['landsat_bias']

    def update_rad_settings(self, band_position):

        # Landsat settings
        if self.sensor.lower() in self.landsat_sensors:

            # Is there a user provided file or object?
            if isinstance(self.metadata, str) or isinstance(self.metadata, raster_tools.LandsatParser):

                # A value of 999 means coefficients
                #   were not gathered.
                coeff_check = self.pr.no_coeff

                # Landsat 8 information pulled from metadata.
                if raster_tools.SENSOR_DICT[self.pr.sensor.lower()] == 'oli_tirs':

                    landsat_gain = self.pr.rad_mult_dict[int(band_position)]
                    landsat_bias = self.pr.rad_add_dict[int(band_position)]

                else:

                    l_max = self.pr.LMAX_dict[int(band_position)]
                    l_min = self.pr.LMIN_dict[int(band_position)]

                    landsat_gain, landsat_bias = self.get_gain_bias(self.pr.series,
                                                                    raster_tools.SENSOR_DICT[self.pr.sensor.lower()],
                                                                    band_position, l_max, l_min, coeff_check)

                self.rad_settings['landsat_gain'] = landsat_gain
                self.rad_settings['landsat_bias'] = landsat_bias

            else:

                bi = self.bands2process[self.bands2process.index(band_position)]

                self.rad_settings['landsat_gain'] = self.landsat_gain_list[bi]
                self.rad_settings['landsat_bias'] = self.landsat_bias_list[bi]

    def scale_data(self, calibrated_array):

        if self.d_type == 'byte':

            return rescale_intensity(calibrated_array,
                                     in_range=(0., 1.),
                                     out_range=(0, 255)).astype(np.uint8)

        elif self.d_type == 'uint16':

            return rescale_intensity(calibrated_array,
                                     in_range=(0., 1.),
                                     out_range=(0, 10000)).astype(np.uint16)

    def create_output(self):

        if isinstance(self.bands2process, int) and self.bands2process == -1:
            self.bands2process = list(range(1, self.i_info.bands+1))
        elif isinstance(self.bands2process, int) and self.bands2process > 0:
            self.bands2process = [self.bands2process]
        elif isinstance(self.bands2process, int) and self.bands2process == 0:
            raise ValueError('\nThe bands to process must be -1, int > 0, or a list of bands.\n')

        self.band_range = list(range(1, len(self.bands2process)+1))

        # Copy the input information.
        self.o_info = self.i_info.copy()

        # Change parameters if necessary.
        self.o_info.storage = self.d_type

        # Create the output.
        self.out_rst = raster_tools.create_raster(self.output_image, self.o_info)

    def get_metadata(self):

        if isinstance(self.metadata, str):

            if self.sensor.lower() in self.landsat_sensors:

                if isinstance(self.metadata, str):
                    self.pr = raster_tools.LandsatParser(self.metadata)

        elif isinstance(self.metadata, raster_tools.LandsatParser):
            self.pr = self.metadata

        if isinstance(self.image_date, str):
            year, month, day = self.image_date.split('/')
        else:
            year, month, day = self.pr.year, self.pr.month, self.pr.day

        if not isinstance(self.solar_angle, float):
            self.solar_angle = self.pr.elev

        self.julian_day = date2julian(month, day, year)


# def search_worldview_meta(dir, type):
#
#     fName = '.IMD'
#
#     # search for date and elevation angle
#     absCalLine = 'absCalFactor'
#     effBandwLine = 'effectiveBandwidth'
#     dateLine = 'earliestAcqTime'
#     eleve_line = 'meanSunEl'
#     group = 'BEGIN_GROUP = BAND'
#
#     absCalFactor, effBandWidth, bandNames = [], [], []
#
#     for root, dirs, files in os.walk(dir):
#
#         if root[-3:] == type:
#             print 'Walking through: %s' % root
#
#             rootList = os.listdir(root)
#
#             for file in rootList:
#
#                 if fName in file:
#
#                     #   : open the text file
#                     txtFile = '%s/%s' % (root, file)
#                     txt     = open(txtFile,'r')
#
#                     #   : get lines in a list
#                     txtRead = txt.readlines()
#
#                     for line in txtRead:
#
#                         if group in line:
#                             #   : get each band
#                             bandNames.append(string.strip(line[line.find('= ')+1:-1]))
#
#                         if absCalLine in line:
#                             #   : get absolute calibration factor for each band
#                             absCalFactor.append(float(line[line.find('= ')+1:-2]))
#
#                         if effBandwLine in line:
#                             #   : get effective bandwidth for each band
#                             effBandWidth.append(float(line[line.find('= ')+1:-2]))
#
#                         if dateLine in line:
#                             #   : get the date
#                             date = string.strip(line[line.find('= ')+1:line.find('= ')+12])
#
#                         if eleve_line in line:
#                             #   : get the elevation angle
#                             elev = float(line[line.find('= ')+1:-2])
#
#     return date, elev, absCalFactor, effBandWidth, bandNames


def get_scan_angle(img=None, i_info=None, band2open=1, forward=True):

    """
    Gets the scan angle from an input Landsat image.

    1) get the bottom left corner, moving right, extracting each column
    2) stop if the column has data (i.e., it's a corner)
    3) get the <y> position where there is data in the column
    4) get the top left corner, moving down, extracting each row
    5) stop if the row has data
    6) get the <x> position where there is data in the row
    7) find the degree of the angle

    scan angle (a) = rad2deg[tangent(opposite / adjacent)]
        where,
            opposite = urx - llx
            adjacent = lly - ury

                         /| urx | ury
                        / |
                       /  |
                      /   |
                     /    | opposite
                    /     |
                   /      |
             lly b/a\_____| lrx
                  adjacent
                  
    Args:
        img (Optional[str]): The input image. Default is None.
        i_info (Optional[object]): An instance of ``raster_tools.ropen``. Default is None.
        band2open (Optional[int]): The band to open. Default is 1.
        foward (Optional[bool]): Whether to rotate the image forward. Default is True.
        
    Returns:
        The rotation degree.
    """

    if i_info:
        img = i_info.read(bands2open=band2open)
    else:
        if not isinstance(img, np.ndarray):
            raise TypeError('\nThe image needs to be given as an array if not given by an ropen instance.\n')

    rws, cls = img.shape

    # get the bottom left corner
    for cl in range(0, cls):

        # all rows, current column
        clm = img[:, cl]

        if clm.max() > 0:
            break

    llx = copy(cl)
    lly_idx = clm[np.where(clm > 0)]

    # first pixel (North to South) that has data
    # try:
    lly = list(clm).index(lly_idx[0])
    # except:
    #     print list(clm)
    #     print
    #     print lly_idx
    #     print
    #     print img.max()
    #     import matplotlib.pyplot as plt
    #     plt.imshow(img)
    #     plt.show()
    #     sys.exit()

    # get the upper right corner
    for rw in range(0, rws):

        # all columns, current row
        rww = img[rw, :]

        if rww.max() > 0:
            break

    ury = copy(rw)
    urx_idx = rww[np.where(rww > 0)]

    # first pixel (West to East) that has data
    urx = list(rww).index(urx_idx[0])

    opposite = float(lly - ury)
    adjacent = float(urx - llx)

    if (opposite == 0) and (adjacent == 0):
        deg = 0
    else:
        # get the rotation degree
        if forward:
            deg = -(90. - np.rad2deg(np.arctan(opposite / adjacent)))
        else:
            deg = 90. - np.rad2deg(np.arctan(opposite / adjacent))

    return deg
