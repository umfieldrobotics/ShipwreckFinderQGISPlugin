# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from osgeo import gdal
from qgisplugin.core.tiff_utils import (crop_tiff,
                                        create_chunks,
                                        merge_chunks,
                                        get_tiff_size,
                                        merge_transparent_parts, 
                                        copy_tiff_metadata,
                                        ensure_valid_nodata,
                                        robust_remove_invalid_pixels,
                                        get_raster_resolution,
                                        remove_small_contours_chunked,
                                        generate_csv)
from qgisplugin.core.train import unet_test, hrnet_test, basnet_test

from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject, QgsPointXY, QgsRasterLayer, QgsRectangle
# from qgis.core import Qgis, QgsProviderRegistry, QgsMapLayerProxyModel, QgsRasterLayer, QgsProject, QgsReferencedRectangle
from osgeo import gdal, osr, gdalconst
import rasterio

import random
import shutil

BASNET_PATH = "/home/smitd/DrewShipwreckSeeker/ShipwreckSeekerQGISPlugin/qgisplugin/core/basnet_ruby-river-240_best.pt" # BEST BASNet
HRNET_PATH = "/home/smitd/DrewShipwreckSeeker/ShipwreckSeekerQGISPlugin/qgisplugin/core/hrnet_splendid-tree-238_best.pt" # BEST HRNet
UNET_PATH = "/home/smitd/DrewShipwreckSeeker/ShipwreckSeekerQGISPlugin/qgisplugin/core/unet_valiant-spaceship-247_best.pt" # BEST UNet
UNETAUX_PATH = "/home/smitd/DrewShipwreckSeeker/ShipwreckSeekerQGISPlugin/qgisplugin/core/unetaux_rosy-elevator-246_best.pt" # BEST UNet Aux

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

    def execute(self, output_path, save_model_output = False, model_arch = 'UNet Hillshade', set_progress: callable = None,
                log: callable = print):
        """
        The core of the plugin
        """

        # Output path for chunks
        set_progress(1)

        # Export the raster as a geotiff
        geotiff_path = os.path.join(self.temp_dir, "exported_geotiff.tif")
        export_raster_as_geotiff(self.raster_layer, geotiff_path)
        set_progress(5)

        res_x, res_y = get_raster_resolution(geotiff_path)

        #Crop the image using the extent
        cropped_path = os.path.join(self.temp_dir, "cropped_geotiff.tif")
        self.crop_image_using_extent(geotiff_path, self.extent_str, cropped_path) # Todo, use this instead
        set_progress(10)

        # Ensure nodata values are valid
        ensure_valid_nodata(cropped_path, cropped_path)

        width, height = get_tiff_size(cropped_path)
        set_progress(15)
  

        chunk_size = int(200 / abs(res_x))
    
        rows, cols = create_chunks(cropped_path, self.temp_dir, chunk_size)


        ignore_images = [cropped_path, geotiff_path]
        input_files = glob.glob(os.path.join(self.temp_dir, "*"))


        for file in input_files:
            if file in ignore_images:
                continue
            with rasterio.open(file) as src:
                array = src.read(1)

            array = array.astype(np.float64)
            output_file_path = os.path.splitext(file)[0] + ".npy"
            np.save(output_file_path, array)            

        set_progress(20)



        if model_arch == "HRNet":
            hrnet_test(self.temp_dir, ignore_images, HRNET_PATH, chunk_size, res_x, set_progress)
        elif model_arch == "BASNet":
            basnet_test(self.temp_dir, ignore_images, BASNET_PATH, chunk_size, res_x, set_progress)
        else:
            if model_arch == "UNet":
                unet_test(self.temp_dir, ignore_images, UNET_PATH, chunk_size, res_x, set_progress, False)
            else:
                unet_test(self.temp_dir, ignore_images, UNETAUX_PATH, chunk_size, res_x, set_progress, True)

        set_progress(80)

        # Merge the chunks
        merge_chunks(self.temp_dir, rows, cols, output_path, save_model_output)


        set_progress(85)
        # Crop back to original size, maintaining geospatial information
        crop_tiff(output_path, output_path, width, height)

        set_progress(90)

        merge_transparent_parts(cropped_path, output_path, output_path)

        invalid_pixels = robust_remove_invalid_pixels(cropped_path, output_path, output_path)

        # Remove small contours
        contour_threshold = 0 # 0.0006 # ------------- THIS IS THE ADJUSTABLE THRESHOLD THAT WILL BECOME A PARAMETER
        remove_small_contours_chunked(output_path, output_path, contour_threshold, invalid_pixels)

        copy_tiff_metadata(cropped_path, output_path)
        set_progress(95)

        output_csv = output_path.replace(".tif", "_ship_locs.csv")
        generate_csv(output_path, output_csv)



def printProgress(value: int):
    """ Replacement for the GUI progress bar """

    print('progress: {} %'.format(value))
