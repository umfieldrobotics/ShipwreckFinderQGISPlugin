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
from ..safe_libs_setup import setup_libs, safe_import_ml_libraries

setup_libs()
libs = safe_import_ml_libraries()

import os
import sys
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
from qgis.PyQt.QtGui import QPixmap, QImage


import os

from qgisplugin.core.ship_seeker import ShipSeeker
from qgisplugin.interfaces import import_image, write_image
from qgisplugin.interfaces.RectangleMapTool import RectangleMapTool
from qgisplugin.core.tiff_utils import convert_png_to_tiff, copy_tiff_metadata
import matplotlib.pyplot as plt
from scipy.special import softmax

class Thresholder:
    def __init__(self, npy_pred):
        self.probabilities = softmax(npy_pred, axis=1)
        # print(self.probabilities[0, :10, :10])

    def get_thresholded_image(self, thresh_value):
        pred_binary = (self.probabilities[:, 1, :, :] > thresh_value).astype(np.uint8)
        return pred_binary


class ThresholdingWidget(QDialog):
    """ QDialog to interactively set up the Neural Network input and output. """

    def __init__(self):
        super(ThresholdingWidget, self).__init__()
        loadUi(op.join(op.dirname(__file__), 'thresholding_gui.ui'), self)

        self.thresholder = None

        self.numpy_file_widget.lineEdit().setPlaceholderText(f"Input raw segmentation model predictions (*.npy)")

        self.original_segmentation_textbox.lineEdit().setReadOnly(True)
        self.original_segmentation_textbox.lineEdit().setPlaceholderText(f"Original Tiff Segmentation Image (*.tif)")
        self.original_segmentation_textbox.setFilter("Tiff (*.tif);;All (*.*)")

        self.outputFileWidget.lineEdit().setReadOnly(True)
        self.outputFileWidget.lineEdit().setPlaceholderText(f"Output tiff file path (*.tif)")
        self.outputFileWidget.setStorageMode(QgsFileWidget.SaveFile)
        self.outputFileWidget.setFilter("Tiff (*.tif);;All (*.*)")

        self.percentageSlider.valueChanged.connect(self._update_percentage_display)
        self.percentageDisplay.textChanged.connect(self._update_percentage_slider)

        # Connect buttons
        self.previewButton.clicked.connect(self._preview_image)
        self.saveButton.clicked.connect(self._save_image)

        self.success_label.setText("")


    def log(self, text):
        # append text to log window
        self.logBrowser.append(str(text) + '\n')
        # open the widget on the log screen
        self.tabWidget.setCurrentIndex(self.tabWidget.indexOf(self.tab_log))

    def _preview_image(self):
        thresholded_image = self.create_image()

        if thresholded_image is not None:
            plt.imsave("tmp.png", thresholded_image, cmap="jet")
            pixmap = QPixmap("tmp.png")
            self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), aspectRatioMode=Qt.KeepAspectRatio))
            os.remove("tmp.png") 

    def _save_image(self):
        thresholded_image = self.create_image()
        output_file_path = self.outputFileWidget.filePath()
        original_semgnation_path = self.original_segmentation_textbox.filePath()

        if thresholded_image is not None and output_file_path != "" and original_semgnation_path != "":
            plt.imsave("tmp.png", thresholded_image, cmap="jet")
            pixmap = QPixmap("tmp.png")
            self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), aspectRatioMode=Qt.KeepAspectRatio))
            convert_png_to_tiff("tmp.png", output_file_path)
            copy_tiff_metadata(original_semgnation_path, output_file_path)
            os.remove("tmp.png")

            self.success_label.setText(f"Image saved to {output_file_path}!")

        if self.openInQGISBox.isChecked():
            output_raster_layer = QgsRasterLayer(output_file_path, 'New Thresholded Image')
            QgsProject.instance().addMapLayer(output_raster_layer, True)

    def create_image(self):
        # Read the value of the input 
        try:
            threshold_value = float(self.percentageDisplay.text().replace("%", ""))/100
        except:
            threshold_value = 0

        input_numpy_path = self.numpy_file_widget.filePath()
        output_file_path = self.outputFileWidget.filePath()

        if input_numpy_path != "":
            np_arr = np.load(input_numpy_path)

            self.thresholder = Thresholder(np_arr)
            pred_thresh = self.thresholder.get_thresholded_image(threshold_value)
            pred_thresh = np.squeeze(pred_thresh)

            return pred_thresh

    def _update_percentage_display(self, value):
        self.percentageDisplay.blockSignals(True)
        self.percentageDisplay.setText(f'{value}%')
        self.percentageDisplay.blockSignals(False)
    
    def _update_percentage_slider(self):
        try:
            threshold_value = float(self.percentageDisplay.text().replace("%", ""))
            print(threshold_value, self.percentageSlider.minimum(), self.percentageSlider.maximum())

            if threshold_value < self.percentageSlider.minimum():
                threshold_value = self.percentageSlider.minimum()
            elif threshold_value > self.percentageSlider.maximum():
                threshold_value = self.percentageSlider.maximum()

            # Update the slider's value
        except:
            threshold_value = 0
            
        self.percentageSlider.blockSignals(True)
        self.percentageSlider.setValue(int(threshold_value))
        self.percentageSlider.blockSignals(False)

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
