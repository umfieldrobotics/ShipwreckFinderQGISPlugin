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
    QDialog,
    QDialogButtonBox,
)
from qgis.PyQt.QtGui import QPixmap

from qgisplugin.core.preprocessing_handler import PreprocessingHandler
from qgisplugin.core.tiff_utils import copy_tiff_metadata


import matplotlib.pyplot as plt
import cv2



class PreprocessingWidget(QDialog):
    """ QDialog to interactively set up the Neural Network input and output. """

    def __init__(self):
        super(PreprocessingWidget, self).__init__()
        loadUi(op.join(op.dirname(__file__), 'preprocessing_gui.ui'), self)
        
        # Image Selection
        excluded_providers = [p for p in QgsProviderRegistry.instance().providerList() if p not in ['gdal']]
        self.imageDropDown.setExcludedProviders(excluded_providers)
        self.imageDropDown.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.imageDropDown.layerChanged.connect(self._choose_image)
        self.imageAction.triggered.connect(self._browse_for_image)
        self.imageButton.setDefaultAction(self.imageAction)

        # Output file
        self.outputFileWidget.lineEdit().setReadOnly(True)
        self.outputFileWidget.lineEdit().setPlaceholderText(f"Output tiff file path...")
        self.outputFileWidget.setStorageMode(QgsFileWidget.SaveFile)
        self.outputFileWidget.setFilter("Tiff (*.tif);;All (*.*)")
        self.outputFileWidget.fileChanged.connect(self.on_output_file_selected)

        # Run button
        self.OKClose.button(QDialogButtonBox.Ok).setText("Run")
        self.OKClose.accepted.connect(self._run)
        self.OKClose.rejected.connect(self.close)

        # self.thresholder = None

        # self.numpy_file_widget.lineEdit().setPlaceholderText(f"Input raw segmentation model predictions (*.npy)")

        # self.outputFileWidget.lineEdit().setReadOnly(True)
        # self.outputFileWidget.lineEdit().setPlaceholderText(f"Output segmentation image file path (*.png)")
        # self.outputFileWidget.setStorageMode(QgsFileWidget.SaveFile)
        # self.outputFileWidget.setFilter("PNG (*.png);;All (*.*)")

        # self.percentageSlider.valueChanged.connect(self._update_percentage_display)

        # self.success_label.setText("")


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
        
    def on_output_file_selected(self):
        self.output_tiff_path = self.outputFileWidget.filePath()


    def _run(self):
        """ Read all parameters and pass them on to the core function. """

        # todo: read all parameters, throw errors when needed, give user feedback and run code
        raster_layer = self.imageDropDown.currentLayer()
        raster_path = raster_layer.dataProvider().dataSourceUri()

        output_path = self.outputFileWidget.filePath()
        
        # Parse the depth array
        self.progressBar.setValue(10)
        url = raster_path.split('|')[0]
        options = raster_path.split('|')[1:]
        if options:
            options[0].replace("option:", "")
            raster_ds = gdal.OpenEx(url, open_options=options)
        else:
            raster_ds = gdal.Open(url)
        depth_band = raster_ds.GetRasterBand(1)
        depth_array = depth_band.ReadAsArray()
        scan_xres = abs(raster_ds.GetGeoTransform()[1])
        scan_yres = abs(raster_ds.GetGeoTransform()[5])
        if (scan_xres != scan_yres):
            print("Scan x and y resolution are not the same")

        self.progressBar.setValue(20)

        # Normalize the depth array
        preprocessor = PreprocessingHandler()
        ker_size_value = self.kernelSizeBox.value() # Taken as User input from the GUI, default 50
        inpaint_rad_value = self.infillRadiusBox.value()
        result_arr, mask = preprocessor.normalize2(depth_array, self.progressBar.setValue, 
                                            kernel_size=ker_size_value, inpaint_radius=inpaint_rad_value,
                                            x_res=scan_xres, y_res=scan_yres)
        
        # Save the result arr to the output_path and copy over the meta data...
        print("Output path:", output_path)
        np.save(output_path.replace('.tif','.npy'), result_arr) # Masked, 200x200x3, between 0 and 1
        # np.save(output_path.replace('.tif','_mask.npy'), mask) # Mask of valid pixels before inpainting, not currently used
        cv2.imwrite(output_path, result_arr)
        copy_tiff_metadata(raster_path, output_path)

        if self.openCheckBox.isChecked():
                output_raster_layer = QgsRasterLayer(output_path, 'Preprocessed Raster')
                QgsProject.instance().addMapLayer(output_raster_layer, True)

        self.progressBar.setValue(100)


def _run():
    """ To run your GUI stand alone: """
    from qgis.core import QgsApplication
    app = QgsApplication([], True)
    app.initQgis()

    z = PreprocessingWidget()
    z.show()

    app.exec_()


if __name__ == '__main__':
    _run()
