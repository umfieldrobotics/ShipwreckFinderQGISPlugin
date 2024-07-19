# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from osgeo import gdal
from qgisplugin.core.tiff_utils import crop_tiff, create_chunks, merge_chunks, get_tiff_size, merge_transparent_parts, copy_tiff_metadata, linear_interpolate_transparent
from qgisplugin.core.train import test
from qgisplugin.core.RasterChunkHandler import RasterChunkHandler


from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject, QgsPointXY, QgsRasterLayer, QgsRectangle
# from qgis.core import Qgis, QgsProviderRegistry, QgsMapLayerProxyModel, QgsRasterLayer, QgsProject, QgsReferencedRectangle
from osgeo import gdal, osr


import random

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

    def crop_image_using_extent(self, image_path, extent_string, output_path):
        # Parsing the input string for extent and source CRS
        extent_str, projection_str = extent_string.split(' ')
        xmin, xmax, ymin, ymax = map(float, extent_str.split(','))
        src_crs = projection_str.strip('[]')

        print("Translating between", src_crs, "to", "EPSG:4326")

        print(xmin, xmax, ymin, ymax)
        gdal_translate_options = gdal.TranslateOptions(
                                    projWin=[xmin, ymax, xmax, ymin],
                                    projWinSRS=src_crs,
                                    outputSRS="EPSG:4326",
                                    format="GTiff"
                                )
        
        input_ds = gdal.Open(image_path)
        gdal.Translate(output_path, input_ds, options=gdal_translate_options)

    #TODO: do this
    def __init__(self, raster_layer, extent_str):
        """
        todo: describe your variables
        """
        self.raster_layer = raster_layer
        self.extent_str = extent_str

    # def load_raster_by_band_name(self, band_str="elevation", chunk_size=1024):

    #     band_index = None
    #     for i in range(1, self.raster_layer.bandCount() + 1):
    #         full_band_name = self.raster_layer.bandName(i)
    #         if band_str in full_band_name:
    #             band_index = i
    #             break

    #     if band_index is None:
    #         raise ValueError(f"No band named '{band_str}' found in the raster")

    #     gdal_dataset = gdal.Open(self.raster_layer.dataProvider().dataSourceUri())
    
    #     if not gdal_dataset:
    #         raise ValueError("Unable to open raster data source")

    #     # Get raster dimensions
    #     width = gdal_dataset.RasterXSize
    #     height = gdal_dataset.RasterYSize

    #     # Initialize an empty list to store chunks
    #     all_chunks = []
    #     band = gdal_dataset.GetRasterBand(band_index)
        
    #     for i in range(0, height, chunk_size):
    #         for j in range(0, width, chunk_size):
    #             # Define the size of the chunk
    #             xoff = j
    #             yoff = i
    #             xsize = min(chunk_size, width - j)
    #             ysize = min(chunk_size, height - i)
                
    #             # Read the chunk into a numpy array
    #             chunk = band.ReadAsArray(xoff, yoff, xsize, ysize)
    #             all_chunks.append(chunk)
        
    #     # Stack all chunks into a single numpy array
    #     # Assume chunks are read row-wise and stacked vertically
    #     numpy_array = np.vstack([np.hstack(row) for row in zip(*[iter(all_chunks)]*((width + chunk_size - 1) // chunk_size))])
        
    #     return numpy_array

    def execute(self, output_path, save_model_output = False, set_progress: callable = None,
                log: callable = print):
        """
        The core of the plugin
        """

        # Output path for chunks
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_chunks")
        os.makedirs(temp_dir, exist_ok=True)
        set_progress(1)

        

        # # Print out the numpy array to determine 
        # print("This is the number of bands: ", self.raster_layer.bandCount())
        # elevation_arr = self.load_raster_by_band_name("elevation")
        # print("Size is ", elevation_arr.size)
        # print("min and max ", np.min(elevation_arr), np.max(elevation_arr))
        chunker = RasterChunkHandler(self.raster_layer.dataProvider().dataSourceUri(), 501)
        print("Chunker size is ", chunker.get_chunk_as_np_arr(0, 0).size)
        print("Chunker arr is ", chunker.get_chunk_as_np_arr(0, 0))
        arr = chunker.get_chunk_as_np_arr(0, 0)
        print("Min in this chunk is ", np.min(arr))
        print("max in this chunk is ", np.max(arr))
        print("Chunker size x is ", chunker.get_num_x_chunks())
        print("Chunker size y is ", chunker.get_num_y_chunks())

        # Export the raster as a geotiff
        geotiff_path = os.path.join(temp_dir, "exported_geotiff.tif")
        export_raster_as_geotiff(self.raster_layer, geotiff_path)

        #Crop the image using the extent
        cropped_path = os.path.join(temp_dir, "cropped_geotiff.tif")
        self.crop_image_using_extent(geotiff_path, self.extent_str, cropped_path) # Todo, use this instead


        # TODO: Just Testing...
        interpolated_path = os.path.join(temp_dir, "cropped_interpolated_geotiff.tif")
        linear_interpolate_transparent(cropped_path, interpolated_path)

        width, height = get_tiff_size(interpolated_path)
        
        # Create cropped images in /temp_chunks/
        rows, cols = create_chunks(interpolated_path, temp_dir)
        ignore_images = [cropped_path, interpolated_path, geotiff_path]

        input_files = glob.glob(os.path.join(temp_dir, "*"))

        # Copy metadata to model output
        for i, input_file_path in enumerate(input_files):
            if input_file_path in ignore_images:
                continue

            output_tiff_file_path = test([input_file_path], WEIGHTS_PATH)[0][0]

            
            copy_tiff_metadata(input_file_path, output_tiff_file_path)

            set_progress(int(100.0*i/len(input_files)))


        # Merge the chunks
        merge_chunks(temp_dir, rows, cols, output_path, save_model_output)
        crop_tiff(output_path, output_path, width, height)

        merge_transparent_parts(cropped_path, output_path, output_path)
        copy_tiff_metadata(cropped_path, output_path)

        # # Clean up all of the chunks
        # for f in os.listdir(temp_dir):
        #     os.remove(os.path.join(temp_dir, f))
        # os.rmdir(temp_dir)

def printProgress(value: int):
    """ Replacement for the GUI progress bar """

    print('progress: {} %'.format(value))
