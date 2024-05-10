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
from os.path import join, dirname, abspath, basename
from qgis.core import QgsRasterLayer, QgsProject
from qgis.PyQt.QtCore import QTimer
from qgisplugin.interfaces.my_gui import MyWidget
from qgisplugin.interfaces import import_image
from tests import ExtendedUnitTesting, app

DATA_FOLDER = join(dirname(abspath(__file__)), "data")


class TestWidget(ExtendedUnitTesting):

    def test_app_opens(self):

        widget = MyWidget()
        widget.show()

        QTimer.singleShot(2000, app.closeAllWindows)
        app.exec_()

    def test_app_runs(self):
        image = join(DATA_FOLDER, "image.tif")
        output = join(DATA_FOLDER, "output.tif")

        layer = QgsRasterLayer(image, basename(image), 'gdal')
        QgsProject.instance().addMapLayer(layer, True)

        widget = MyWidget()

        # filling out items
        widget.imageDropDown.setLayer(layer)
        widget.normalizationCheckBox.setChecked(True)
        widget.outputFileWidget.lineEdit().setText(output)

        # clicking OK
        widget._run()

        result, _ = import_image(output)
        self.assertEqual(result.shape[0], layer.bandCount())
        self.assertEqual(result.shape[1], layer.height())
        self.assertEqual(result.shape[2], layer.width())

        # clean up
        QgsProject.instance().removeAllMapLayers()
        self.clean_up([output, '{}.aux.xml'.format(image)])
