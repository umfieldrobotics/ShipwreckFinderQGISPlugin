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
import os.path as op
import numpy as np
import tempfile

from osgeo import gdal
from qgis.gui import QgsFileWidget
from qgis.core import QgsProviderRegistry, QgsMapLayerProxyModel, QgsRasterLayer, QgsProject
from qgis.utils import iface
from qgis.PyQt.uic import loadUi
from qgis.PyQt.QtWidgets import QDialog, QFileDialog, QDialogButtonBox


from qgisplugin.core.my_code import MyCode
from qgisplugin.interfaces import import_image, write_image


class MyWidget(QDialog):
    """ QDialog to interactively set up the Neural Network input and output. """

    def __init__(self):
        super(MyWidget, self).__init__()
        loadUi(op.join(op.dirname(__file__), 'my_gui.ui'), self)

        # todo: link widgets to code in your __init__ function

        # input
        excluded_providers = [p for p in QgsProviderRegistry.instance().providerList() if p not in ['gdal']]
        self.imageDropDown.setExcludedProviders(excluded_providers)
        self.imageDropDown.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.imageDropDown.layerChanged.connect(self._choose_image)
        self.imageAction.triggered.connect(self._browse_for_image)
        self.imageButton.setDefaultAction(self.imageAction)

        # parameters
        self.normalizationCheckBox.stateChanged.connect(self._toggle_normalization)

        # output
        self.outputFileWidget.lineEdit().setReadOnly(True)
        self.outputFileWidget.lineEdit().setPlaceholderText('[Drew we are testing]')
        self.outputFileWidget.setStorageMode(QgsFileWidget.SaveFile)
        self.outputFileWidget.setFilter("Tiff (*.tif);;All (*.*)")

        # Open in QGIS?
        try:
            iface.activeLayer
        except AttributeError:
            self.openCheckBox.setChecked(False)
            self.openCheckBox.setDisabled(True)

        # run or cancel
        self.OKClose.button(QDialogButtonBox.Ok).setText("Run")
        self.OKClose.accepted.connect(self._run)
        self.OKClose.rejected.connect(self.close)

        # widget variables
        self.image = None
        self.classified = None

    def log(self, text):
        # append text to log window
        self.logBrowser.append(str(text) + '\n')
        # open the widget on the log screen
        self.tabWidget.setCurrentIndex(self.tabWidget.indexOf(self.tab_log))

    def _browse_for_image(self):
        """ Browse for an image raster file. """

        path = QFileDialog.getOpenFileName(filter=QgsProviderRegistry.instance().fileRasterFilters())[0]

        try:
            if len(path) > 0:
                gdal.UseExceptions()
                layer = QgsRasterLayer(path, op.basename(path), 'gdal')
                assert layer.isValid()
                QgsProject.instance().addMapLayer(layer, True)

                self.imageDropDown.setLayer(layer)

        except AssertionError:
            self.log("'" + path + "' not recognized as a supported file format.")
        except Exception as e:
            self.log(e)

    def _toggle_normalization(self, state: int):
        """ Enable the normalization spin box when checking the box. """

        self.normalizationSpinBox.setEnabled(state)

    def _choose_image(self):
        """ When the user browsers for an image, the normalization value is set automatically. """

        layer = self.imageDropDown.currentLayer()

        if layer is None:
            return
        try:
            block = layer.dataProvider().block(1, layer.extent(), layer.width(), layer.height())
            row = np.repeat(np.arange(0, layer.height()), layer.width())
            col = np.tile(np.arange(0, layer.width()), layer.height())
            data = [block.value(x, y) for x, y in zip(row, col)]

            # a short list of unique values
            limit = np.nanmax(np.array(data))
            self.normalizationSpinBox.setValue(limit)

        except Exception as e:
            self.log(e)

    def _run(self):
        """ Read all parameters and pass them on to the core function. """

        # todo: read all parameters, throw errors when needed, give user feedback and run code

        try:
            # Only temp file possible when result is opened in QGIS
            output_path = self.outputFileWidget.filePath()

            if not self.openCheckBox.isChecked() and len(output_path) == 0:
                raise Exception("If you won't open the result in QGIS, you must select a base file name for output.")

            # Get parameters
            image_path = self.imageDropDown.currentLayer().source()
            image, metadata = import_image(image_path)

            check = self.normalizationCheckBox.isEnabled()
            quotient = self.normalizationSpinBox.value() if check else None
            add = self.constantSpinBox.value()
            threshold = self.thresholdSpinBox.value()

            # run code
            result = MyCode(image=image, normalize=check, quotient=quotient)\
                .execute(constant=add, threshold=threshold, set_progress=self.progressBar.setValue, log=self.log)
            result = result * quotient if check else result

            self.progressBar.setValue(100)

            # write image to file
            if len(output_path) == 0:
                output_path = op.join(tempfile.gettempdir(), op.basename(op.splitext(image_path)[0]))

            output_path = write_image(file_path=output_path, image=result, geo_transform=metadata['geo_transform'],
                                      projection=metadata['projection'])

            # Open result in QGIS
            if self.openCheckBox.isChecked():
                output_raster_layer = QgsRasterLayer(output_path, 'New Image')
                QgsProject.instance().addMapLayer(output_raster_layer, True)

        except AttributeError:
            self.log("Please select an image.")
        except Exception as e:
            self.log(e)


def _run():
    """ To run your GUI stand alone: """
    from qgis.core import QgsApplication
    app = QgsApplication([], True)
    app.initQgis()

    z = MyWidget()
    z.show()

    app.exec_()


if __name__ == '__main__':
    _run()
