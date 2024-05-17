# -*- coding: utf-8 -*-

import numpy as np
import torch
import os
from osgeo import gdal

import random

OUTPUT_PATH = "/home/frog/dev/output/out.tif"
LAYER_NAME = "Mesa"

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

def create_chunks(input_path, output_dir, chunk_size=512):
    # Open the raster dataset
    dataset = gdal.Open(input_path)
    if dataset is None:
        raise Exception("Failed to open the raster dataset.")

    # Get raster properties
    num_bands = dataset.RasterCount
    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize

    # Calculate number of chunks
    x_chunks = (x_size + chunk_size - 1) // chunk_size
    y_chunks = (y_size + chunk_size - 1) // chunk_size

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each chunk
    for i in range(x_chunks):
        for j in range(y_chunks):
            # Calculate chunk coordinates and size
            x_offset = i * chunk_size
            y_offset = j * chunk_size
            width = min(chunk_size, x_size - x_offset)
            height = min(chunk_size, y_size - y_offset)

            # Create chunk file name
            chunk_filename = f"chunk_{i}_{j}.tif"
            chunk_path = os.path.join(output_dir, chunk_filename)

            # Create chunk dataset
            driver = gdal.GetDriverByName("GTiff")
            chunk_dataset = driver.Create(chunk_path, width, height, num_bands, dataset.GetRasterBand(1).DataType)

            # Copy geotransform and projection from original dataset
            chunk_dataset.SetGeoTransform((
                dataset.GetGeoTransform()[0] + x_offset * dataset.GetGeoTransform()[1],
                dataset.GetGeoTransform()[1],
                0,
                dataset.GetGeoTransform()[3] + y_offset * dataset.GetGeoTransform()[5],
                0,
                dataset.GetGeoTransform()[5]
            ))
            chunk_dataset.SetProjection(dataset.GetProjection())

            # Read and write data for each band
            for band_num in range(1, num_bands + 1):
                band = dataset.GetRasterBand(band_num)
                chunk_band = chunk_dataset.GetRasterBand(band_num)



                data = band.ReadAsArray(x_offset, y_offset, width, height)

                k = random.uniform(0.5, 1.5)
                data = data * k 
                data[data>255] = 255

                chunk_band.WriteArray(data)

            # Close chunk dataset
            chunk_dataset = None

    # Close original dataset
    dataset = None

def merge_chunks(output_dir, output_path):
    # Get a list of chunk files
    chunk_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.tif')]
    
    # Use gdal_merge.py script to merge the chunks
    gdal_merge_cmd = f"gdal_merge.py -o {output_path} " + " ".join(chunk_files)
    os.system(gdal_merge_cmd)

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

    def add_to_image(self, constant: float) -> np.ndarray:
        """
        Add a constant to an image.

        :param constant: The constant to add to each pixel of the image.
        :return: The new image.
        """

        return self.image + constant

    def execute(self, set_progress: callable = None,
                log: callable = print):
        """
        The core of the plugin
        """

        # Output path for chunks
        temp_dir = os.path.join(os.path.dirname(OUTPUT_PATH), "temp_chunks")
        os.makedirs(temp_dir, exist_ok=True)

        


        # Export the raster as a geotiff
        geotiff_path = os.path.join(temp_dir, "exported_geotiff.tif")
        export_raster_as_geotiff(self.raster_layer, geotiff_path)


        # Create all of the chunks 
        create_chunks(geotiff_path, temp_dir)

        # Merge all of the chunks
        merge_chunks(temp_dir, OUTPUT_PATH)

        # Clean up all of the chunks
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)

def printProgress(value: int):
    """ Replacement for the GUI progress bar """

    print('progress: {} %'.format(value))
