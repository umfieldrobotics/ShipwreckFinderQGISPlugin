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
# todo: update all of these variables with your own information
dense_name = 'python-package'  # used for distributing your package on pip or qgis, no spaces
long_name = 'Shipwreck Seeker Plugin'  # what users see in the QGIS plugin repository and on the RTD toc
pdf_title = 'Shipwreck Seeker Plugin'  # the front page title on the read the docs PDF version

author = 'Drew'  # your name
author_email = 'drewskis@umich.edu'  # your contact information
# author_copyright = '© 2020 by Ann Crabbé'  # a copyright, typical "© [start year] - [end year] by [your name]"
short_version = '1.0'  # 2 numbers, update with each new release
long_version = '1.0.1'  # 3 numbers, update with each new release

bitbucket_home = 'https://www.google.com'  # home page of bitbucket
bitbucket_src = 'https://www.google.com'  # src page of bitbucket
bitbucket_issues = 'https://www.google.com'  # issue page of bitbucket

read_the_docs = 'https://www.google.com'  # read the docs page (check URL is free)

keywords = ['LUMOS', 'QGIS', 'plugin', 'BELSPO', 'STEREO']  # to help others find your plugin in the repository

qgis_min_version = '3.6'  # modern plugins will typically break below 3.4

short_description = 'Project to help the user set up a QGIS plugin'
long_description = 'Project to help the user set up a QGIS plugin, along with documentation, ' \
                   'a python package and unit testing.'

qgis_metadata_icon = 'images/plugin_logo.png'
# icon must be square, used in the qgis plugin repository list, QGIS menu, processing toolbox
qgis_category = 'Raster'  # in which QGIS menu does this plugin belong? typically raster for remote sensing plugins
processing_provider = True  # change to False if you do not foresee a processing toolbox item
