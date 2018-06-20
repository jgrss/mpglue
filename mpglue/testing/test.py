#!/usr/bin/env python

import os
import sys
import subprocess

from ..paths import get_main_path


def main():

    version_info = sys.version_info

    if version_info.major == 2:
        py = 'python'
    else:
        py = 'python3'

    subprocess.call('{PY} {COM}'.format(PY=py,
                                        COM=os.path.join(get_main_path(), 'testing', 'testing.py')),
                    shell=True)


if __name__ == '__main__':
    main()
