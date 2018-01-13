#!/usr/bin/env python

import os
import sys
import argparse


def vrt_updater(input_vrt, search_string, replace_string):

    """
    Updates a VRT file

    Args:
        input_vrt (str): The input VRT file.
        search_string (str): A string to search for and replace.
        replace_string (str): A string to replace `search_string`.

    Returns:
        None, writes to `input_vrt`.
    """

    with open(input_vrt, 'rb') as vrt_file:

        vrt_lines = vrt_file.readlines()

        for vi, vrt_line in enumerate(vrt_lines):
            vrt_lines[vi] = vrt_line.replace(search_string, replace_string)

    # Remove the VRT file.
    os.remove(input_vrt)

    # Write the updated lines.
    with open(input_vrt, 'w') as vrt_file:
        vrt_file.writelines(vrt_lines)


def _examples():

    sys.exit("""\

    vrt_updater.py -i /vrt_file.vrt -ss /old_string -rs /new_string

    """)


def main():

    parser = argparse.ArgumentParser(description='Updates a VRT file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-i', '--input', dest='input', help='The input VRT file', default=None)
    parser.add_argument('-ss', '--search-string', dest='search_string',
                        help='A string to search for and replace', default=None)
    parser.add_argument('-rs', '--replace-string', dest='replace_string',
                        help='A string to replace -ss', default=None)

    args = parser.parse_args()

    if args.examples:
        _examples()

    vrt_updater(args.input,
                args.search_string,
                args.replace_string)


if __name__ == '__main__':
    main()
