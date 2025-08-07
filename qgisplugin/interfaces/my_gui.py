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

from qgis.PyQt.QtWidgets import (
    QMenu,
    QAction,
    QDialog,
    QVBoxLayout,
    QDialogButtonBox,
    QLabel,
    QComboBox
)
from qgis.PyQt.QtCore import QCoreApplication, pyqtSignal
from qgis.PyQt.QtGui import QCursor


from qgisplugin.core.ship_seeker import ShipSeeker
from qgisplugin.interfaces import import_image, write_image
from qgisplugin.interfaces.RectangleMapTool import RectangleMapTool


class LayerSelectionDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr('Select Extent'))

        vl = QVBoxLayout()
        vl.addWidget(QLabel(self.tr('Use extent from')))
        self.combo = QgsMapLayerComboBox()
        self.combo.setFilters(
            Qgis.LayerFilter.HasGeometry | Qgis.LayerFilter.RasterLayer | Qgis.LayerFilter.MeshLayer)
        vl.addWidget(self.combo)

        self.button_box = QDialogButtonBox()
        self.button_box.setStandardButtons(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        vl.addWidget(self.button_box)
        self.setLayout(vl)

    def selected_layer(self):
        return self.combo.currentLayer()

class MyWidget(QDialog):
    """ QDialog to interactively set up the Neural Network input and output. """

    def __init__(self):
        super(MyWidget, self).__init__()
        loadUi(op.join(op.dirname(__file__), 'my_gui.ui'), self)

        # todo: link widgets to code in your __init__ function

        if iface is not None:
            canvas = iface.mapCanvas()
            self.prevMapTool = canvas.mapTool()
            self.tool = RectangleMapTool(canvas)
            self.tool.rectangleCreated.connect(self.updateExtent)
        else:
            self.prevMapTool = None
            self.tool = None

        # input
        excluded_providers = [p for p in QgsProviderRegistry.instance().providerList() if p not in ['gdal']]
        self.imageDropDown.setExcludedProviders(excluded_providers)
        self.imageDropDown.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.imageDropDown.layerChanged.connect(self._choose_image)
        self.imageAction.triggered.connect(self._browse_for_image)
        self.imageButton.setDefaultAction(self.imageAction)

        self.extentButton.clicked.connect(self.selectExtent)

        # other parameters
        self.percentageSlider.valueChanged.connect(self.updateSliderPercent)

        # output
        self.outputFileWidget.lineEdit().setReadOnly(True)
        self.outputFileWidget.lineEdit().setPlaceholderText(f"Output tiff file path...")
        self.outputFileWidget.setStorageMode(QgsFileWidget.SaveFile)
        self.outputFileWidget.setFilter("Tiff (*.tif);;All (*.*)")
        self.outputFileWidget.fileChanged.connect(self.on_output_file_selected)

        # Open in QGIS?
        try:
            iface.activeLayer
        except AttributeError:
            self.openCheckBox.setChecked(False)
            self.openCheckBox.setDisabled(True)

        # Save model output?
        self.savePredictionCheckBox.setChecked(False)

        # run or cancel
        self.OKClose.button(QDialogButtonBox.Ok).setText("Run")
        self.OKClose.accepted.connect(self._run)
        self.OKClose.rejected.connect(self.close)

        # widget variables
        self.image = None
        self.classified = None

    def on_output_file_selected(self):
        output_tiff_path = self.outputFileWidget.filePath()

        output_npy_path = op.splitext(output_tiff_path)[0] + ".npy"

        self.savePredictionCheckBox.setText(f"Save model prediction array ({output_npy_path})")


    def setExtentValueFromRect(self, r):
        s = '{},{},{},{}'.format(
            r.xMinimum(), r.xMaximum(), r.yMinimum(), r.yMaximum())

        try:
            self.crs = r.crs()
        except:
            self.crs = QgsProject.instance().crs()
        if self.crs.isValid():
            s += ' [' + self.crs.authid() + ']'

        self.extentText.setText(s)
        self.tool.reset()
        canvas = iface.mapCanvas()
        canvas.setMapTool(self.prevMapTool)
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def selectOnCanvas(self):
        canvas = iface.mapCanvas()
        canvas.setMapTool(self.tool)
        self.showMinimized()

    def useLayerExtent(self):
        dlg = LayerSelectionDialog(self)
        if dlg.exec():
            layer = dlg.selected_layer()
            self.setExtentValueFromRect(QgsReferencedRectangle(layer.extent(), layer.crs()))
    
    def useCanvasExtent(self):
        self.setExtentValueFromRect(QgsReferencedRectangle(iface.mapCanvas().extent(), 
                                                           iface.mapCanvas().mapSettings().destinationCrs()))

    def useCurrentLayerExtent(self):
        layer = self.imageDropDown.currentLayer()
        self.setExtentValueFromRect(QgsReferencedRectangle(layer.extent(), layer.crs()))

    def updateExtent(self):
        r = self.tool.rectangle()
        self.setExtentValueFromRect(r)

    def updateSliderPercent(self, value):
        self.percentageSliderValue.setText(f"{value}%")

    def selectExtent(self):
        popupmenu = QMenu()

        useCanvasExtentAction = QAction(
            QCoreApplication.translate("ExtentSelectionPanel", 'Use Canvas Extent'),
            self.extentButton)
        useLayerExtentAction = QAction(
            QCoreApplication.translate("ExtentSelectionPanel", 'Use Layer Extent…'),
            self.extentButton)
        useCurrentLayerExtentAction = QAction(
            QCoreApplication.translate("ExtentSelectionPanel", "Use Current Layer"),
            self.extentButton)
        selectOnCanvasAction = QAction(
            self.tr('Select Extent on Canvas'), self.extentButton)
        
        popupmenu.addAction(useCanvasExtentAction)
        popupmenu.addAction(useLayerExtentAction)
        popupmenu.addAction(useCurrentLayerExtentAction)
        popupmenu.addSeparator()
        popupmenu.addAction(selectOnCanvasAction)

        selectOnCanvasAction.triggered.connect(self.selectOnCanvas)
        useLayerExtentAction.triggered.connect(self.useLayerExtent)
        useCurrentLayerExtentAction.triggered.connect(self.useCurrentLayerExtent)
        useCanvasExtentAction.triggered.connect(self.useCanvasExtent)

        popupmenu.exec(QCursor.pos())

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

            # image_path = self.imageDropDown.currentLayer().source()
            # image, metadata = import_image(image_path)

            extent_str = self.extentText.text()

            model_arch = self.findChild(QComboBox, "modelDropdown").currentText()

            # run code
            result = ShipSeeker(raster_layer=raster_layer, extent_str=extent_str)\
                .execute(output_path, save_model_output=self.savePredictionCheckBox.isChecked(), model_arch=model_arch, \
                         set_progress=self.progressBar.setValue, log=self.log)

            self.progressBar.setValue(100)

            # write image to file
            # print("The output path will be: ", output_path)

            # Open result in QGIS
            if self.openCheckBox.isChecked():
                output_name = output_path.split('/')[-1][:-4]
                # output_raster_layer = QgsRasterLayer(output_path, 'New Image')
                output_raster_layer = QgsRasterLayer(output_path, output_name)
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

    z = MyWidget()
    z.show()

    app.exec_()


if __name__ == '__main__':
    _run()
