# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from osgeo import gdal
from qgisplugin.core.tiff_utils import crop_tiff, create_chunks, merge_chunks, get_tiff_size, merge_transparent_parts, copy_tiff_metadata
from qgisplugin.core.train import test

import random

OUTPUT_PATH = "/home/frog/dev/output/out.tif"
WEIGHTS_PATH = "/home/frog/dev/ShipwreckSeekerQGISPlugin/qgisplugin/core/mbes_unet.pt"

from qgis.core import (
    QgsRasterFileWriter,
    QgsRasterPipe
)

def export_raster_as_geotiff(raster_layer, output_path):
    # Export the raster as GeoTIFF
    writer = QgsRasterFileWriter(output_path)
    pipe = QgsRasterPipe()
    pipe.set(raster_layer.dataProvider().clone())

    #TODO: This is deprecated
    writer.writeRaster(pipe, raster_layer.width(), raster_layer.height(), raster_layer.extent(), raster_layer.crs())

class ShipSeeker:
    """
    The code responsible for splitting up an input raster image into chunks, converting each to 
    ML format, finding potential shipwrecks using pytorch, then combining the resulting chunks to
    a final raster image. 
    """

    #TODO: do this
    def __init__(self, raster_layer):
        """
        todo: describe your variables

        :param image: e.g. "Input image [n bands x m rows x b bands]"
        :param normalize: e.g. "Set to true to normalize the image"
        :param quotient: e.g. "Normalisation quotient for the image. Ignored if variable_2 is set to False
        """
        self.raster_layer = raster_layer

    def execute(self, output_path, set_progress: callable = None,
                log: callable = print):
        """
        The core of the plugin
        """

        # Output path for chunks
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_chunks")
        os.makedirs(temp_dir, exist_ok=True)
        set_progress(5)


        # Export the raster as a geotiff
        geotiff_path = os.path.join(temp_dir, "exported_geotiff.tif")
        export_raster_as_geotiff(self.raster_layer, geotiff_path)
        width, height = get_tiff_size(geotiff_path)
        
        # Create cropped images in /temp_chunks/
        create_chunks(geotiff_path, temp_dir)
        os.remove(geotiff_path)

        input_files = glob.glob(os.path.join(temp_dir, "*"))

        # Copy metadata to model output
        for i, input_file_path in enumerate(input_files):
            output_file_path = test([input_file_path], WEIGHTS_PATH)[0]

            merge_transparent_parts(input_file_path, output_file_path, output_file_path)
            copy_tiff_metadata(input_file_path, output_file_path)

            set_progress(int(100.0*i/len(input_files)))

        # Merge the chunks
        merge_chunks(temp_dir, output_path)
        crop_tiff(output_path, output_path, width, height)

        # Clean up all of the chunks
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)

def printProgress(value: int):
    """ Replacement for the GUI progress bar """

    print('progress: {} %'.format(value))
