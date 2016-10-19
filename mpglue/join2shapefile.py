#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 9/8/2015
"""

import os
import sys
import time
import shutil
import argparse
import string
import re

from .helpers import shp2dataframe, dataframe2dbf

try:
    import editdist
    editdist_installed = True
except:
    editdist_installed = False

try:
    from fuzzywuzzy import fuzz
    fuzz_installed = True
except:
    fuzz_installed = False

try:
    from num2words import num2words
    num2words_installed = True
except:
    num2words_installed = False

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

reload(sys)
sys.setdefaultencoding('Cp1252')


def _find_digits(string2search, language='es'):

    if not num2words_installed:
        raise ImportError('Num2words must be installed to use score matching.')

    searched_string = filter(str.isdigit, str(string2search))

    if searched_string:

        # replace digits with words
        string2search = string2search.replace(searched_string, str(num2words(int(searched_string), lang=language)))

    return string2search


def _normalize(s, convert_digits=True, language='es'):

    for p in string.punctuation:
        s = s.replace(p, '')

    if convert_digits:
        s = _find_digits(s.lower().strip(), language=language)
    else:
        s = s.lower().strip()

    return s


def _get_total_score(string1, string2):

    if not fuzz_installed:
        raise ImportError('Fuzzywuzzy must be installed to use score matching.')

    lookup = {'0': 0, '2': .2, '3': .3, '4': .4, '5': .5, '6': .6, '7': .7, '8': .8, '9': .9, '1': 1.}

    total_score = (500 - (fuzz.ratio(string1, string2) * 5)) * editdist.distance(string1, string2)
    # total_score = (100 - (fuzz.ratio(string1, string2) * 5)) * editdist.distance(string1, string2)
    # total_score = (100 - fuzz.ratio(string1, string2)) * editdist.distance(string1, string2)

    if string1 == string2:
        total_score -= 5000

    # remove 500 points if one of the departments is in the other
    if (string1 in string2) or (string2 in string1):
        total_score -= 500

    # check for three-character keys
    # and remove 200 points per three-letter character found
    bs = [string1[i:i+3] for i in xrange(len(string1)-2)]
    total_score -= sum([200 for b in bs if re.search('(%s.*)' % b, string2)])

    bs = [string2[i:i+3] for i in xrange(len(string2)-2)]
    total_score -= sum([200 for b in bs if re.search('(%s.*)' % b, string1)])

    # split words
    for s1_split in string1.split(' '):

        for s2_split in string2.split(' '):

            sc = str(fuzz.ratio(s1_split, s2_split))

            if len(sc) == 1:
                sc = lookup['0']
            elif len(sc) == 2:
                sc = np.exp(lookup[sc[0]]) * int(sc)
            else:
                sc = np.exp(lookup['1']) * int(sc)

            total_score -= sc * 5

            if s1_split == s2_split:
                total_score -= 50

    return total_score


def _match_geoid(df_gadm, current_ADM1, current_ADM2, current_ADM3, total_dict, geo_id_total_dict,
                 row_dict, row_index, highest_level, current_ADM0='none', min_adm0='none',
                 encode_adm1s=False, language='es'):

    """
    Matches strings based on scoring metrics
    """

    geo_id = '-999'

    adm0_score_dict = {}
    adm1_score_dict = {}
    adm2_score_dict = {}
    adm3_score_dict = {}
    score_dict = {}
    geo_id_dict = {}

    # if encode_adm1s:
    #     unique_adm1s = [u'%s' % unique_adm1 for unique_adm1 in unique_adm1s]
    #     unique_adm1s = [u'%s' % unique_adm1.encode('ascii', 'ignore') for unique_adm1 in unique_adm1s]

    # GET THE CURRENT ADM0.

    if min_adm0 == 'none':

        # List all of the ADM0s.
        unique_adm0s = np.unique(df_gadm.loc[:, 'ADM0'].values)

        for unique_adm0 in unique_adm0s:

            unique_adm0_norm = _normalize(unique_adm0, language=language)

            total_score = _get_total_score(current_ADM0, unique_adm0_norm)

            adm0_score_dict[unique_adm0] = float('%.2f' % total_score)

        # Get the closest matching adm0.
        min_adm0 = min(adm0_score_dict, key=lambda k: adm0_score_dict[k])

    if highest_level == 'ADM1':

        # GET THE CURRENT ADM1.

        unique_adm1s = df_gadm.loc[df_gadm['ADM0'] == min_adm0, 'ADM1'].unique()

        for unique_adm1 in unique_adm1s:

            unique_adm1_norm = _normalize(unique_adm1, language=language)

            total_score = _get_total_score(current_ADM1, unique_adm1_norm)

            adm1_score_dict[unique_adm1] = float('%.2f' % total_score)

        # Get the closest matching ADM1.
        min_adm1 = min(adm1_score_dict, key=lambda k: adm1_score_dict[k])

        geo_id = df_gadm.loc[(df_gadm['ADM0'] == min_adm0) & (df_gadm['ADM1'] == min_adm1), :].GeoId.values[0]

    elif highest_level == 'ADM2':

        # GET THE CURRENT ADM1.

        unique_adm1s = df_gadm.loc[df_gadm['ADM0'] == min_adm0, 'ADM1'].unique()

        for unique_adm1 in unique_adm1s:

            unique_adm1_norm = _normalize(unique_adm1, language=language)

            total_score = _get_total_score(current_ADM1, unique_adm1_norm)

            adm1_score_dict[unique_adm1] = float('%.2f' % total_score)

        # Get the closest matching adm1.
        min_adm1 = min(adm1_score_dict, key=lambda k: adm1_score_dict[k])

        # GET THE CURRENT ADM2.

        unique_adm2s = df_gadm.loc[(df_gadm['ADM0'] == min_adm0) & (df_gadm['ADM1'] == min_adm1), 'ADM2'].unique()

        for unique_adm2 in unique_adm2s:

            unique_adm2_norm = _normalize(unique_adm2, language=language)

            total_score = _get_total_score(current_ADM2, unique_adm2_norm)

            adm2_score_dict[unique_adm2] = float('%.2f' % total_score)

        # Get the closest matching ADM2.
        min_adm2 = min(adm2_score_dict, key=lambda k: adm2_score_dict[k])

        if not total_dict:
            total_dict['{}-{}-{}'.format(min_adm0, min_adm1, min_adm2)] = adm2_score_dict[min_adm2]
        else:

            # Check if the ADM has already been entered.
            if min_adm2 in total_dict:
                print 'ALERT!!!'
                print total_dict
                print
                print adm1_score_dict[min_adm1]
                print adm2_score_dict[min_adm2]
                sys.exit()

        geo_id = df_gadm.loc[(df_gadm['ADM0'] == min_adm0) & (df_gadm['ADM1'] == min_adm1) & \
                             (df_gadm['ADM2'] == min_adm2), :].GeoId.values[0]

    elif highest_level == 'ADM3':

        # GET THE CURRENT ADM1.

        unique_adm1s = df_gadm.loc[df_gadm['ADM0'] == min_adm0, 'ADM1'].unique()

        for unique_adm1 in unique_adm1s:

            unique_adm1_norm = _normalize(unique_adm1, language=language)

            total_score = _get_total_score(current_ADM1, unique_adm1_norm)

            adm1_score_dict[unique_adm1] = float('%.2f' % total_score)

        # get the closest matching adm1
        min_adm1 = min(adm1_score_dict, key=lambda k: adm1_score_dict[k])

        # GET THE CURRENT ADM2.

        unique_adm2s = df_gadm.loc[(df_gadm['ADM0'] == min_adm0) & (df_gadm['ADM1'] == min_adm1), 'ADM2'].unique()

        for unique_adm2 in unique_adm2s:

            unique_adm2_norm = _normalize(unique_adm2, language=language)

            total_score = _get_total_score(current_ADM2, unique_adm2_norm)

            adm2_score_dict[unique_adm2] = float('%.2f' % total_score)

        # Get the closest matching ADM2.
        min_adm2 = min(adm2_score_dict, key=lambda k: adm2_score_dict[k])

        # GET THE CURRENT ADM3.

        unique_adm3s = df_gadm.loc[(df_gadm['ADM0'] == min_adm0) & (df_gadm['ADM1'] == min_adm1) & \
                                   (df_gadm['ADM2'] == min_adm2), 'ADM3'].unique()

        for unique_adm3 in unique_adm3s:

            unique_adm3_norm = _normalize(unique_adm3, language=language)

            total_score = _get_total_score(current_ADM3, unique_adm3_norm)

            adm3_score_dict[unique_adm3] = float('%.2f' % total_score)

        # Get the closest matching ADM3.
        min_adm3 = min(adm3_score_dict, key=lambda k: adm3_score_dict[k])

        geo_id = df_gadm.loc[(df_gadm['ADM0'] == min_adm0) & (df_gadm['ADM1'] == min_adm1) & \
                             (df_gadm['ADM2'] == min_adm2) & (df_gadm['ADM3'] == min_adm3), :].GeoId.values[0]

    return geo_id, total_dict, geo_id_total_dict, row_dict

    # else:

        # if highest_level == 'ADM3':
        #
        #     adm2_score_dict = {}
        #
        #     # List all of the ADM2s.
        #     unique_adm2s = np.unique(df_gadm.loc[:, 'ADM2'].values)
        #
        #     # Get the current ADM2.
        #     for unique_adm2 in unique_adm2s:
        #
        #         unique_adm2_norm = _normalize(unique_adm2, language=language)
        #
        #         total_score = _get_total_score(current_ADM2, unique_adm2_norm)
        #
        #         adm2_score_dict[unique_adm2] = float('%.2f' % total_score)
        #
        #     # get the closest matching adm2
        #     min_adm2 = min(adm2_score_dict, key=lambda k: adm2_score_dict[k])
        #
        #     df_current_unique_adm_level = df_gadm.loc[df_gadm['ADM2'] == min_adm2, :]
        #
        # else:
        #
        #     # Subset the dataframe to get
        #     # the current ADM1.
        #     df_current_unique_adm_level = df_gadm.loc[df_gadm['ADM1'] == min_adm1, :]
        #
        # # Iterate over all rows to check for the highest level
        # # that is close to matching the current highest level.
        # for df_gadm_sub_index, df_gadm_sub_row in df_current_unique_adm_level.iterrows():
        #
        #     department2compare = _normalize(df_gadm_sub_row[highest_level], language=language)
        #
        #     # Compare the strings.
        #     total_score = _get_total_score(current_highest, department2compare)
        #
        #     # Add the total score and GeoId.
        #     score_dict[department2compare] = float('%.2f' % total_score)
        #     # geo_id_dict[department2compare] = str(df_gadm_sub_row.geo3)
        #     geo_id_dict[department2compare] = str(df_gadm_sub_row.GeoId)
        #
        # if score_dict:
        #
        #     break2999 = False
        #
        #     geo_id_found = False
        #
        #     while not geo_id_found:
        #
        #         # We are looking for the smallest score
        #         # this returns the name of the department with the smallest score
        #         if score_dict:
        #             min_department = min(score_dict, key=lambda k: score_dict[k])
        #         else:
        #             geo_id = '-999'
        #             break2999 = True
        #             break
        #
        #         # check if the province has already been added
        #         if current_ADM1 in total_dict:
        #
        #             # check if the GEO Id has already been used
        #             if geo_id_dict[min_department] in geo_id_total_dict[current_ADM1].values():
        #
        #                 # If they are the same department with a
        #                 # different variable (e.g., Planted, Harvested, ...),
        #                 # then use the same id. No need to update the
        #                 # 'total' dictionaries.
        #                 if min_department in total_dict[current_ADM1]:
        #
        #                     score1 = total_dict[current_ADM1][min_department]
        #                     score2 = score_dict[min_department]
        #
        #                     if score1 == score2:
        #
        #                         geo_id_found = True
        #
        #                     else:
        #
        #                         # if the department already in the dictionary has a smaller score,
        #                         # remove the department from the score dictionary and get the next
        #                         # smallest department
        #                         if score1 < score2:
        #
        #                             del score_dict[min_department]
        #                             del geo_id_dict[min_department]
        #
        #                         # if the new department has a smaller score, swap the ids
        #                         else:
        #
        #                             total_dict[current_ADM1][min_department] = score2
        #                             geo_id_total_dict[current_ADM1][min_department] = geo_id_dict[min_department]
        #
        #                             geo_id_found = True
        #
        #                 # If the lowest ranking department is not
        #                 # already in the 'total' dictionary,
        #                 # but the Geo Id is.
        #                 else:
        #
        #                     # score1 = total_dict[current_ADM1][min_department]
        #                     # score2 = score_dict[min_department]
        #
        #                     print '2'
        #                     print geo_id_total_dict
        #                     print
        #                     print score_dict
        #                     print
        #                     print geo_id_dict
        #                     print
        #                     print current_highest, '--', min_department
        #                     print geo_id_dict[min_department]
        #                     sys.exit()
        #
        #             # if this is the first use of the Geo Id,
        #             # then add the score and Id and break
        #             # out of the loop.
        #             else:
        #
        #                 total_dict[current_ADM1][min_department] = score_dict[min_department]
        #                 geo_id_total_dict[current_ADM1][min_department] = geo_id_dict[min_department]
        #
        #                 geo_id_found = True
        #
        #         # Add the province, score, and Geo Id
        #         # and break out of the loop.
        #         else:
        #
        #             total_dict[current_ADM1] = {}
        #             geo_id_total_dict[current_ADM1] = {}
        #
        #             total_dict[current_ADM1][min_department] = score_dict[min_department]
        #             geo_id_total_dict[current_ADM1][min_department] = geo_id_dict[min_department]
        #
        #             geo_id_found = True
        #
        #     if not break2999:
        #         geo_id = geo_id_dict[min_department]
        #
        # row_dict[row_index] = geo_id
        #
        # return geo_id, total_dict, geo_id_total_dict, row_dict


def _get_highest(dataframe_row, adm0, language, lookup_dict={}):

    current_adm0 = 'none'
    current_adm1 = 'none'
    current_adm2 = 'none'
    current_adm3 = 'none'

    if adm0:
        current_adm0 = _normalize(dataframe_row.ADM0.lower().replace('_', ' '), language=language)

    if 'ADM3' in dataframe_row.keys():

        current_adm1 = _normalize(dataframe_row.ADM1.lower().replace('_', ' '), language=language)
        current_adm2 = _normalize(dataframe_row.ADM2.lower().replace('_', ' '), language=language)
        current_adm3 = _normalize(dataframe_row.ADM3.lower().replace('_', ' '), language=language)

        current_highest = 'ADM3'

    elif 'ADM2' in dataframe_row.keys():

        current_adm1 = _normalize(dataframe_row.ADM1.lower().replace('_', ' '), language=language)
        current_adm2 = _normalize(dataframe_row.ADM2.lower().replace('_', ' '), language=language)

        if lookup_dict:

            if current_adm1 in lookup_dict:

                if current_adm2 in lookup_dict[current_adm1]:
                    current_adm2 = lookup_dict[current_adm1][current_adm2]

        current_highest = 'ADM2'

    elif 'ADM1' in dataframe_row.keys():

        current_adm1 = _normalize(dataframe_row.ADM1.lower().replace('_', ' '), language=language)

        current_highest = 'ADM1'

    return current_highest, current_adm0, current_adm1, current_adm2, current_adm3


def _score_matching(df_, df_gadm, adm0=True, language='es'):

    total_dict = {}
    geo_id_total_dict = {}
    row_dict = {}

    df_ = df_.set_index([range(0, df_.shape[0])])

    # TODO: let user define unique column
    # df_gadm['GeoId'] = range(0, df_gadm.shape[0])

    for row in xrange(0, df_.shape[0]):

        df_row = df_.iloc[row, :]

        current_highest, current_adm0, current_adm1, current_adm2, current_adm3 = _get_highest(df_row, adm0, language)

        if current_highest not in ['ADM1', 'ADM2', 'ADM3']:
            continue

        geo_id, total_dict, geo_id_total_dict, row_dict = _match_geoid(df_gadm, current_adm1, current_adm2,
                                                                       current_adm3, total_dict,
                                                                       geo_id_total_dict, row_dict, row,
                                                                       current_highest, current_ADM0=current_adm0)

        df_.loc[row, 'GeoId'] = geo_id

    return df_


def join2shapefile(shapefile, table, output, merge_fields=[], how2join='inner',
                   shp_field='UNQ', tbl_field='UNQ', score_matching=False,
                   adm0=None, adm1=None, adm2=None, adm3=None, language='en'):

    """
    Args:
        shapefile (str): The shapefile to join to.
        table (str or dataframe): The table or Pandas dataframe to join to ``shapefile``.
        output (str): The output shapefile of ``shapefile`` + ``table``.
        merge_fields (Optional[str list]): A list of fields to concatenate and create the shapefile join field.
            Default is []. E.g., ``merge_fields``=['ADM0', 'UNQ'] would result in values under both to concatenate
            to '[ADM0-value][UNQ-value]'. More specifically, if ADM0=ARG and UNQ=10233, the result would be
            ARG10233. *Note that the new, concatenated field is always named 'GeoId'.
        how2join (Optional[str]): How to join the data. Choices are ['inner', 'outer']. Default is 'inner'.
        shp_field (Optional[str])): The join field for the shapefile. Default is 'UNQ'.
        tbl_field (Optional[str])): The join field for the table. Default is 'UNQ'.
        score_matching (Optional[bool]): Whether to use string score matching to join fields. Default is False.
        adm0 (Optional[dict]): {'tbl': 'ADM0', 'shp': 'COUNTRY'}
        adm1 (Optional[dict]): {'tbl': 'ADM1', 'shp': 'PROVINCE'}
        adm2 (Optional[dict]): {'tbl': 'ADM2', 'shp': 'NAME'}
        language (Optional[str]): The language to convert digits. Default is 'en'. Choices are [en, fr, de, es,
            lt, lv, en_GB, en_IN].

    Examples:
        >>> from mappy.tables import join2shapefile
        >>>
        >>> # Join a table to a shapefile
        >>> join2shapefile('/shapefile.shp', '/data_table.csv', '/output.shp', \
        >>>                shp_field='UNQ', tbl_field='ID')
        >>>
        >>> # Score matching
        >>> join2shapefile('/shapefile.shp', '/data_table.csv', '/output.shp',
        >>>                score_matching=True, adm1='NAME1', adm2='NAME2',
        >>>                shp_field='GeoId', language='en')

    Returns:
        None, writes to ``output``.
    """

    # Open the shapefile DBF and convert to
    # a Pandas dataframe
    d_name, f_name = os.path.split(shapefile)
    f_base, f_ext = os.path.splitext(f_name)

    shp_df = shp2dataframe(shapefile)

    if merge_fields:
        shp_df['GeoId'] = shp_df[merge_fields[0]].astype(str) + shp_df[merge_fields[1]].astype(str)

    # join_df_ = pd.read_table(table, chunksize=1024)
    if isinstance(table, str):
        join_df = pd.read_csv(table, sep=',')
    else:
        join_df = table

    # Ensure the same data type.
    try:
        shp_df[shp_field] = shp_df[shp_field].astype(int).astype(str)
    except:
        shp_df[shp_field] = shp_df[shp_field].astype(str)

    try:
        join_df[tbl_field] = join_df[tbl_field].astype(int).astype(str)
    except:
        join_df[tbl_field] = join_df[tbl_field].astype(str)

    # TODO: add other adm0 and adm3...
    if score_matching:

        if not editdist_installed:
            raise ImportError('Editdist must be installed to use score matching.')

        # tbl_field: 'GeoId'
        join_df.rename(columns={adm0['tbl']: 'ADM0', adm1['tbl']: 'ADM1', adm2['tbl']: 'ADM2'}, \
                       inplace=True)

        shp_df.rename(columns={adm0['shp']: 'ADM0', adm1['shp']: 'ADM1', adm2['shp']: 'ADM2', shp_field: 'GeoId'}, \
                      inplace=True)

        # TODO: ADM level parameters
        df = _score_matching(join_df, shp_df, language=language)

        # Ensure same datatype
        # try:
        #     shp_df[shp_field] = shp_df[shp_field].astype(int)
        # except:
        #     shp_df[shp_field] = shp_df[shp_field].astype(str)

        # try:
        #     df[tbl_field] = df[tbl_field].astype(int)
        # except:
        #     df[tbl_field] = df[tbl_field].astype(str)

        df = pd.merge(shp_df, df, on='GeoId', how=how2join)

    else:
        df = pd.merge(shp_df, join_df, left_on=shp_field, right_on=tbl_field, how=how2join)

    td_name, tf_name = os.path.split(table)
    tf_base, __ = os.path.splitext(tf_name)

    od_name, of_name = os.path.split(output)
    of_base, of_ext = os.path.splitext(of_name)

    # df.to_csv('{}/{}__{}.csv'.format(od_name, of_base, tf_base), sep=',')

    associated_files = os.listdir(d_name)

    for associated_file in associated_files:

        a_base, a_ext = os.path.splitext(associated_file)

        out_file = '{}/{}{}'.format(od_name, of_base, a_ext)

        if a_base == f_base:

            if os.path.isfile(out_file):
                os.remove(out_file)

            if 'dbf' in a_ext.lower():
                dataframe2dbf(df, out_file)
            else:
                shutil.copy2('{}/{}{}'.format(d_name, a_base, a_ext), out_file)


def _examples():

    sys.exit("""\

    # Join both on the UNQ field
    join2shapefile.py -s /join_shapefile.shp -t join_table.csv -o /out_shapefile.shp

    # Concatenate the shapefile fields ADM0 and UNQ into a GeoId field
    join2shapefile.py -s /join_shapefile.shp -t join_table.csv -o /out_shapefile.shp -sf GeoId -tf UNQ -mf ADM0 UNQ

    """)


def main():

    parser = argparse.ArgumentParser(description='Joins a table to a shapefile',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-s', '--shapefile', dest='shapefile', help='The input shapefile to join to', default=None)
    parser.add_argument('-t', '--table', dest='table', help='The input table to join', default=None)
    parser.add_argument('-o', '--output', dest='output', help='The output shapefile', default=None)
    parser.add_argument('-j', '--how2join', dest='how2join', help='How to join the data', default='inner',
                        choices=['innner', 'outer'])
    parser.add_argument('-sf', '--shp_field', dest='shp_field', help='The shapefile field to join on', default='UNQ')
    parser.add_argument('-tf', '--tbl_field', dest='tbl_field', help='The table field to join on', default='UNQ')
    parser.add_argument('-mf', '--merge_fields', dest='merge_fields',
                        help='A list of fields to merge to create the join field', default=[], nargs='+')

    args = parser.parse_args()

    if args.examples:
        _examples()

    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    join2shapefile(args.shapefile, args.table, args.output, how2join=args.how2join,
                   shp_field=args.shp_field, tbl_field=args.tbl_field,
                   merge_fields=args.merge_fields)

    print('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
          (time.asctime(time.localtime(time.time())), (time.time()-start_time)))

if __name__ == '__main__':
    main()
