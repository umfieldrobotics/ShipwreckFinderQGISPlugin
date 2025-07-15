# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from osgeo import gdal
from qgisplugin.core.tiff_utils import crop_tiff, create_chunks, merge_chunks, get_tiff_size, merge_transparent_parts, copy_tiff_metadata, linear_interpolate_transparent
from qgisplugin.core.train import test

from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject, QgsPointXY, QgsRasterLayer, QgsRectangle
# from qgis.core import Qgis, QgsProviderRegistry, QgsMapLayerProxyModel, QgsRasterLayer, QgsProject, QgsReferencedRectangle
from osgeo import gdal, osr, gdalconst
import rasterio

import random
import shutil


# WEIGHTS_PATH = "/home/smitd/DrewShipwreckSeeker/ShipwreckSeekerQGISPlugin/qgisplugin/core/mbes_unet.pt" # Original
WEIGHTS_PATH = "/home/smitd/DrewShipwreckSeeker/ShipwreckSeekerQGISPlugin/qgisplugin/core/unet_lemon-oath-149_best.pt" # New one channel
# WEIGHTS_PATH = "/home/smitd/DrewShipwreckSeeker/ShipwreckSeekerQGISPlugin/qgisplugin/core/unet_aux_effortless-dust-150_best.pt" # New hillshade

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
        
        input_ds = gdal.Open(image_path, gdalconst.GA_ReadOnly)

        proj = osr.SpatialReference(wkt=input_ds.GetProjection())
        proj.AutoIdentifyEPSG()
        epsg_id = proj.GetAttrValue('AUTHORITY',1)

        gdal_translate_options = gdal.TranslateOptions(
                                    projWin=[xmin, ymax, xmax, ymin],
                                    projWinSRS=src_crs,
                                    outputSRS=f"EPSG:{epsg_id}",
                                    format="GTiff"
                                )
        
    # geo_trans = tif_with_RPCs.GetGeoTransform()
    # tif_without_RPCs.SetGeoTransform(geo_trans)
    # tif_without_RPCs.SetProjection(tif_with_RPCs.GetProjection())
        gdal.Translate(output_path, input_ds, options=gdal_translate_options)

    #TODO: do this
    def __init__(self, raster_layer, extent_str):
        """
        todo: describe your variables
        """
        self.raster_layer = raster_layer
        self.extent_str = extent_str

        self.remove_tmp = True
        self.temp_dir = os.path.join("/tmp", "SHIPWRECK_SEEKER", "temp_chunks")
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        os.makedirs(self.temp_dir, exist_ok=False)

    def __del__(self):
        if self.remove_tmp and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def execute(self, output_path, save_model_output = False, set_progress: callable = None,
                log: callable = print):
        """
        The core of the plugin
        """

        # Output path for chunks
        set_progress(1)

        # Export the raster as a geotiff
        geotiff_path = os.path.join(self.temp_dir, "exported_geotiff.tif")
        export_raster_as_geotiff(self.raster_layer, geotiff_path)

        #Crop the image using the extent
        cropped_path = os.path.join(self.temp_dir, "cropped_geotiff.tif")
        self.crop_image_using_extent(geotiff_path, self.extent_str, cropped_path) # Todo, use this instead

        # TODO: Just Testing...
        interpolated_path = os.path.join(self.temp_dir, "cropped_interpolated_geotiff.tif")
        # linear_interpolate_transparent(cropped_path, interpolated_path)

        # width, height = get_tiff_size(interpolated_path)
        width, height = get_tiff_size(cropped_path)

        print(f"Width, Height = {width}, {height}")

        # TODO: Just Thinking...
        # Check size of geotiff here, if it's above a certain size, break into X number
        # of pieces and run evaluation on each of those (still breaking into their own chunks)
        # and then piece back together at the end
        
        # Create cropped images in /temp_chunks/
        # rows, cols = create_chunks(interpolated_path, self.temp_dir)
        rows, cols = create_chunks(cropped_path, self.temp_dir)

        ignore_images = [cropped_path, interpolated_path, geotiff_path]

        input_files = glob.glob(os.path.join(self.temp_dir, "*"))

        for file in input_files:
            if file in ignore_images:
                continue
            with rasterio.open(file) as src:
                array = src.read(1)

            print(f"Creating npy, dtype: {array.dtype}")
            array = array.astype(np.float64)
            output_file_path = os.path.splitext(file)[0] + ".npy"
            np.save(output_file_path, array)

        # Printing all files in tempdir
        # for file in os.listdir(self.temp_dir):
        #     print(file)

        # print(f"Ignore images: {ignore_images}")

        # print(f"Temp dir: {self.temp_dir}")

        print("About to run test function")

        test(self.temp_dir, ignore_images, WEIGHTS_PATH)

        print("Finished running test function")

        # # Copy metadata to model output
        # for i, input_file_path in enumerate(input_files):
        #     if input_file_path in ignore_images:
        #         continue

        #     output_tiff_file_path = test([input_file_path], WEIGHTS_PATH)[0][0]

        #     # output_tiff_file_path = test(self.temp_dir, ignore_images, WEIGHTS_PATH)[0][0]
            
            
        #     copy_tiff_metadata(input_file_path, output_tiff_file_path)

        #     set_progress(int(100.0*i/len(input_files)))

        # print("Finished outer testing loop")
        # print(f"Output path: {output_path}")

        # Merge the chunks
        merge_chunks(self.temp_dir, rows, cols, output_path, save_model_output)

        # print("Just merged chunks")
        # import time
        # time.sleep(120)

        # print(f"Output path after merge: {output_path}")

        print("SeekerHere1")
        # print(f"Tiff width, height: {get_tiff_size(output_path)}")
        # print(f"Cropping down to ({width}, {height})")

        crop_tiff(output_path, output_path, width, height)

        print("SeekerHere2")

        merge_transparent_parts(cropped_path, output_path, output_path)

        print("SeekerHere3")

        copy_tiff_metadata(cropped_path, output_path)

        print("SeekerHere4")

def printProgress(value: int):
    """ Replacement for the GUI progress bar """

    print('progress: {} %'.format(value))
