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

from osgeo import gdal, osr
from qgis.gui import QgsFileWidget, QgsMapLayerComboBox
from qgis.core import Qgis, QgsProviderRegistry, QgsMapLayerProxyModel, QgsRasterLayer,QgsVectorLayer, QgsProject, QgsReferencedRectangle, QgsSymbol, QgsFillSymbol, QgsRendererCategory, QgsCategorizedSymbolRenderer

from qgis.utils import iface
from qgis.PyQt.uic import loadUi
from qgis.PyQt.QtWidgets import QDialog, QFileDialog, QDialogButtonBox

from qgis.PyQt.QtCore import Qt

from qgis.PyQt.QtWidgets import (
    QDialog,
    QDialogButtonBox,
)
from qgis.PyQt.QtGui import QPixmap, QColor

from qgisplugin.core.preprocessing_handler import PreprocessingHandler
from qgisplugin.core.tiff_utils import copy_tiff_metadata
from qgisplugin.core.extract_bb import BoundingBoxExtractor

import fiona
from fiona.crs import from_epsg
from shapely.geometry import Polygon, mapping

import matplotlib.pyplot as plt
import cv2


class ExtractBoxesWidget(QDialog):
    """ QDialog to interactively set up the Neural Network input and output. """

    def __init__(self):
        super(ExtractBoxesWidget, self).__init__()
        loadUi(op.join(op.dirname(__file__), 'extract_shipwrecks_gui.ui'), self)
        
        # Image Selection
        excluded_providers = [p for p in QgsProviderRegistry.instance().providerList() if p not in ['gdal']]
        self.imageDropDown.setExcludedProviders(excluded_providers)
        self.imageDropDown.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.imageDropDown.layerChanged.connect(self._choose_image)
        self.imageAction.triggered.connect(self._browse_for_image)
        self.imageButton.setDefaultAction(self.imageAction)

        # Output file
        self.outputFileWidget.lineEdit().setReadOnly(True)
        self.outputFileWidget.lineEdit().setPlaceholderText(f"Output bounding box vector file path...")
        self.outputFileWidget.setStorageMode(QgsFileWidget.SaveFile)
        self.outputFileWidget.setFilter("Shapefiles (*.shp);;All Files (*.*)")
        self.outputFileWidget.fileChanged.connect(self.on_output_file_selected)

        # Run button
        self.OKClose.button(QDialogButtonBox.Ok).setText("Run")
        self.OKClose.accepted.connect(self._run)
        self.OKClose.rejected.connect(self.close)

        


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

    def raster_to_numpy(self, raster_ds):
        num_bands = raster_ds.RasterCount
    
        bands = []
        
        for i in range(1, num_bands + 1):
            band = raster_ds.GetRasterBand(i)            
            band_array = band.ReadAsArray()
            bands.append(band_array)
        
        array_3d = np.stack(bands, axis=-1)
        
        return array_3d
    
    def export_shape_file(self, bounding_boxes_coords, crs_epsg, output_path):
        from qgis.core import (QgsField, QgsFields, QgsFeature, 
                          QgsGeometry, QgsPointXY, QgsVectorFileWriter,
                          QgsCoordinateReferenceSystem, QgsWkbTypes)
        from qgis.PyQt.QtCore import QVariant

        # Define the fields for the shapefile
        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        fields.append(QgsField("bbox_id", QVariant.String))
        
        # Create coordinate reference system
        crs = QgsCoordinateReferenceSystem(f"EPSG:{crs_epsg}")
        
        # Set up the vector file writer
        writer = QgsVectorFileWriter(
            output_path,
            "UTF-8",
            fields,
            QgsWkbTypes.Polygon,
            crs,
            "ESRI Shapefile"
        )
        
        print("Export shape 2 - Writing features to shapefile")
        
        # Process each bounding box
        for idx, bbox_coords in enumerate(bounding_boxes_coords):
            # Ensure the polygon is closed by checking if first and last points are the same
            coords_list = list(bbox_coords)
            if len(coords_list) > 0 and coords_list[0] != coords_list[-1]:
                coords_list.append(coords_list[0])  # Close the polygon
            
            # Convert coordinates to QgsPointXY objects
            qgs_points = [QgsPointXY(float(x), float(y)) for x, y in coords_list]
            
            # Create polygon geometry
            polygon_geom = QgsGeometry.fromPolygonXY([qgs_points])
            
            # Create feature
            feature = QgsFeature()
            feature.setGeometry(polygon_geom)
            feature.setAttributes([idx + 1, f"bbox_{idx + 1}"])
            
            # Add feature to the writer
            writer.addFeature(feature)
        
        # Delete the writer to ensure all data is written
        del writer
        
        print(f"Shapefile created successfully at: {output_path}")
        
        ## Export to vector layer
        if self.openCheckBox.isChecked():
            output_shapefile_layer = QgsVectorLayer(output_path, 'Bounding Boxes', 'ogr')
                
            QgsProject.instance().addMapLayer(output_shapefile_layer, True)

            print("Export shape 3")

            symbol = QgsSymbol.defaultSymbol(output_shapefile_layer.geometryType())
        
            # Set the color with transparency (e.g., 50% transparent red)
            color = QColor(255, 0, 0, 128)  # Red with 50% transparency
            symbol.setColor(color)
            
            # Apply the symbol to the layer's renderer
            output_shapefile_layer.renderer().setSymbol(symbol)
            
            print("Export shape 4")

            # Refresh the layer to see the changes
            output_shapefile_layer.triggerRepaint()
            print("Export shape 5")
    



    def _run(self):
        """ Read all parameters and pass them on to the core function. """

        raster_layer = self.imageDropDown.currentLayer()
        raster_path = raster_layer.dataProvider().dataSourceUri()
        output_path = self.outputFileWidget.filePath()

        
        # Parse the image of tiff
        self.progressBar.setValue(10)
        url = raster_path.split('|')[0]
        options = raster_path.split('|')[1:]
        if options:
            options[0].replace("option:", "")
            raster_ds = gdal.OpenEx(url, open_options=options)
        else:
            raster_ds = gdal.Open(url)

        np_segmentation_image = self.raster_to_numpy(raster_ds)


        # Get the CRS of the TIFF
        projection = raster_ds.GetProjection()
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromWkt(projection)
        crs_epsg = spatial_ref.GetAttrValue('AUTHORITY', 1)  # EPSG code
        geotransform = raster_ds.GetGeoTransform()

        self.progressBar.setValue(20)

        # Extract the bounding boxes
        bb_extractor = BoundingBoxExtractor(np_segmentation_image, geotransform)
        red_mask = bb_extractor.get_mask((127, 0, 0, 255))
        bounding_boxes_coords = bb_extractor.extract_metric_bb(red_mask, 
                                                               self.minThresholdBox.value(), 
                                                               self.maxThresholdBox.value())

        self.progressBar.setValue(80)

        # Write the shape file
        self.export_shape_file(bounding_boxes_coords, crs_epsg, output_path)
        self.progressBar.setValue(100)

        return

def _run():
    """ To run your GUI stand alone: """
    from qgis.core import QgsApplication
    app = QgsApplication([], True)
    app.initQgis()

    z = ExtractBoxesWidget()
    z.show()

    app.exec_()


if __name__ == '__main__':
    _run()
