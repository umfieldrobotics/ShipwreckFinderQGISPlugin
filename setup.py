# -*- coding: utf-8 -*-
"""
| ----------------------------------------------------------------------------------------------------------------------
| Date                : October 2020                                 todo: change date, copyright and email in all files
| Copyright           : © 2020 by Ann Crabbé
| Email               : acrabbe.foss@gmail.com
| Acknowledgements    : Based on 'Create A QGIS Plugin' [https://bitbucket.org/kul-reseco/create-qgis-plugin]
|                       Crabbé Ann and Somers Ben; funded by BELSPO STEREO III (Project LUMOS - SR/01/321)
|
| This file is part of the [INSERT PLUGIN NAME] plugin and [INSERT PYTHON PACKAGE NAME] python package.
| todo: make sure to fill out your plugin name and python package name here.
|
| This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
| License as published by the Free Software Foundation, either version 3 of the License, or any later version.
|
| This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
| warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
|
| You should have received a copy of the GNU General Public License (COPYING.txt). If not see www.gnu.org/licenses.
| ----------------------------------------------------------------------------------------------------------------------
"""
from os import path
from setuptools import setup, find_packages
from package_variables import *

# read the contents of your README file
with open(path.join(path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    read_me = f.read()

setup(
    name=dense_name,
    version=long_version,

    description=long_description,
    description_content_type='text/x-rst',
    long_description=read_me,
    long_description_content_type='text/x-rst',
    keywords=keywords,

    author=author,
    author_email=author_email,

    url=bitbucket_home,
    project_urls={
        'Documentation': read_the_docs,
        'Source Code': bitbucket_src,
        'Issue Tracker': bitbucket_issues,
    },

    packages=find_packages(exclude=['*test*']),
    data_files=[("", ["COPYING.txt", "package_variables.py", "requirements.txt"])],
    include_package_data=True,
    zip_save=False,

    # todo: change endpoint name for command line interface:
    entry_points={
        'console_scripts': [
            'mycli=qgisplugin.interfaces.my_cli:main'
        ]
    },

    classifier='License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)'
)
