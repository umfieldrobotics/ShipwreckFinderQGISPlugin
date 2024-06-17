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
from qgis.gui import QgsFileWidget, QgsMapLayerComboBox
from qgis.core import Qgis, QgsProviderRegistry, QgsMapLayerProxyModel, QgsRasterLayer, QgsProject, QgsReferencedRectangle

from qgis.utils import iface
from qgis.PyQt.uic import loadUi
from qgis.PyQt.QtWidgets import QDialog, QFileDialog, QDialogButtonBox

from qgis.PyQt.QtCore import Qt

from qgis.PyQt.QtWidgets import (
    QMenu,
    QAction,
    QDialog,
    QVBoxLayout,
    QDialogButtonBox,
    QLabel,
    QApplication,
    QPushButton
)
from qgis.PyQt.QtCore import QCoreApplication, pyqtSignal
from qgis.PyQt.QtGui import QCursor
from qgis.PyQt.QtGui import QPixmap


from qgisplugin.core.ship_seeker import ShipSeeker
from qgisplugin.interfaces import import_image, write_image
from qgisplugin.interfaces.RectangleMapTool import RectangleMapTool

import matplotlib.pyplot as plt

class Drewpers:
    def __init__(self, npy_pred):
        self.npy_pred = npy_pred

    def get_thresholded_image(self, thresh_value):
        pred_binary = (self.npy_pred > thresh_value).astype(np.uint8)
        pred_binary = pred_binary.argmax(axis=1)
        return pred_binary


# class LayerSelectionDialog(QDialog):

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle(self.tr('Select Extent'))

#         vl = QVBoxLayout()
#         vl.addWidget(QLabel(self.tr('Use extent from')))
#         self.combo = QgsMapLayerComboBox()
#         self.combo.setFilters(
#             Qgis.LayerFilter.HasGeometry | Qgis.LayerFilter.RasterLayer | Qgis.LayerFilter.MeshLayer)
#         vl.addWidget(self.combo)

#         self.button_box = QDialogButtonBox()
#         self.button_box.setStandardButtons(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
#         self.button_box.accepted.connect(self.accept)
#         self.button_box.rejected.connect(self.reject)

#         vl.addWidget(self.button_box)
#         self.setLayout(vl)

#     def selected_layer(self):
#         return self.combo.currentLayer()

class ThresholdingWidget(QDialog):
    """ QDialog to interactively set up the Neural Network input and output. """

    def __init__(self):
        super(ThresholdingWidget, self).__init__()
        loadUi(op.join(op.dirname(__file__), 'thresholding_gui.ui'), self)

        self.thresholder = Drewpers(np.zeros(5))

        self.numpy_file_widget.lineEdit().setPlaceholderText(f"Input raw segmentation model predictions (*.npy)")

        self.outputFileWidget.lineEdit().setReadOnly(True)
        self.outputFileWidget.lineEdit().setPlaceholderText(f"Output segmentation image file path (*.png)")
        self.outputFileWidget.setStorageMode(QgsFileWidget.SaveFile)
        self.outputFileWidget.setFilter("PNG (*.png);;All (*.*)")

        self.percentageSlider.valueChanged.connect(self._update_percentage_display)

        self.success_label.setText("")


    def log(self, text):
        # append text to log window
        self.logBrowser.append(str(text) + '\n')
        # open the widget on the log screen
        self.tabWidget.setCurrentIndex(self.tabWidget.indexOf(self.tab_log))

    def _update_percentage_display(self, value):
        self.percentageDisplay.setText(f'{value}%')

        input_numpy_path = self.numpy_file_widget.filePath()
        output_file_path = self.outputFileWidget.filePath()

        if input_numpy_path != "" and output_file_path != "":
            np_arr = np.load(input_numpy_path)

            self.thresholder = Drewpers(np_arr)
            pred_thresh = self.thresholder.get_thresholded_image(value/100)
            pred_thresh = np.squeeze(pred_thresh)

            plt.imsave(output_file_path, pred_thresh, cmap="jet")
            pixmap = QPixmap(output_file_path)
            self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), aspectRatioMode=Qt.KeepAspectRatio))

            self.success_label.setText(f"Image saved to {output_file_path}!")


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

                pixmap = QPixmap(path)
                self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), aspectRatioMode=Qt.KeepAspectRatio))

        # except AssertionError:
        #     self.log("'" + path + "' not recognized as a supported file format.")
        except Exception as e:
            self.log(e)
            raise e

    def _choose_image(self):
        """ When the user browsers for an image """

        layer = self.imageDropDown.currentLayer()

        if layer is None:
            return

    def _run(self):
        """ Read all parameters and pass them on to the core function. """

        # todo: read all parameters, throw errors when needed, give user feedback and run code

        try:
            # Only temp file possible when result is opened in QGIS
            output_path = self.outputFileWidget.filePath()

            if not self.openCheckBox.isChecked() and len(output_path) == 0:
                raise Exception("If you won't open the result in QGIS, you must select a base file name for output.")

            # Get parameters
            raster_layer = self.imageDropDown.currentLayer()

            image_path = self.imageDropDown.currentLayer().source()
            image, metadata = import_image(image_path)

            extent_str = self.extentText.text()
            print("Just got the text...")

            # run code
            result = ShipSeeker(raster_layer=raster_layer, extent_str=extent_str)\
                .execute(output_path, set_progress=self.progressBar.setValue, log=self.log)

            self.progressBar.setValue(100)

            # write image to file
            print("THe output path will be: ", output_path)

            # Open result in QGIS
            if self.openCheckBox.isChecked():
                output_raster_layer = QgsRasterLayer(output_path, 'New Image')
                QgsProject.instance().addMapLayer(output_raster_layer, True)

        # except AttributeError:
        #     self.log("Please select an image.")
        except Exception as e:
            self.log(e)
            raise e


def _run():
    """ To run your GUI stand alone: """
    from qgis.core import QgsApplication
    app = QgsApplication([], True)
    app.initQgis()

    z = ThresholdingWidget()
    z.show()

    app.exec_()


if __name__ == '__main__':
    _run()
