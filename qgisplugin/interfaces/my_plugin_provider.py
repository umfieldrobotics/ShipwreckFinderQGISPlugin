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
from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon
from qgisplugin.interfaces.my_plugin_processing import MyProcessingAlgorithm


class MyProcessingProvider(QgsProcessingProvider):

    def loadAlgorithms(self, *args, **kwargs):
        self.addAlgorithm(MyProcessingAlgorithm())

    def id(self, *args, **kwargs):
        """ The ID of your plugin, used for identifying the provider. This string should be a unique, short,
        character only string, eg "qgis" or "gdal". This string should not be localised. """
        return 'my_plugin_provider'

    def name(self, *args, **kwargs):
        """ The human friendly name of your plugin in Processing. This string should be as short as possible. """
        return self.tr('My Plugin Section')

    def icon(self):
        """ Should return a QIcon which is used for your provider inside the Processing toolbox. """
        return QIcon(':/plugin_logo')
