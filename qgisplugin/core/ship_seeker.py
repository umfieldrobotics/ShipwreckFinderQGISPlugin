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
from qgisplugin.core.utils import clear_directory


import random
import shutil

# WEIGHTS_PATH = "/home/frog/dev/ShipwreckSeekerQGISPlugin/qgisplugin/core/mbes_unet.pt"
WEIGHTS_PATH = "/home/tylergs/Documents/noaa_multibeam_real_data/Training/Models/cnp_data_5_latest.pt"
# WEIGHTS_PATH = "/home/tylergs/Documents/noaa_multibeam_real_data/Training/mbes_unet.pt"

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

        self.chunk_size = 200 # This should be in meters, but we need to scale it by res still

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

        # This directory is just needed for testing, TODO: remove this line
        clear_directory("/home/tylergs/Documents/noaa_multibeam_real_data/Training/Plugin_outputs")

        # Adding test to see if we can extract raster res info
        raster_path = self.raster_layer.dataProvider().dataSourceUri()
        url = raster_path.split('|')[0]
        options = raster_path.split('|')[1:]
        if options:
            options[0].replace("option:", "")
            raster_ds = gdal.OpenEx(url, open_options=options)
        else:
            raster_ds = gdal.Open(url)
        scan_xres = abs(raster_ds.GetGeoTransform()[1])
        scan_yres = abs(raster_ds.GetGeoTransform()[5])
        # print("Testing, raster x/y res", scan_xres, scan_yres) # these are right

        # Export the raster as a geotiff
        geotiff_path = os.path.join(self.temp_dir, "exported_geotiff.tif")
        export_raster_as_geotiff(self.raster_layer, geotiff_path)

        #Crop the image using the extent
        cropped_path = os.path.join(self.temp_dir, "cropped_geotiff.tif")
        self.crop_image_using_extent(geotiff_path, self.extent_str, cropped_path) # Todo, use this instead


        # TODO: Just Testing...
        interpolated_path = os.path.join(self.temp_dir, "cropped_interpolated_geotiff.tif")
        linear_interpolate_transparent(cropped_path, interpolated_path)

        width, height = get_tiff_size(interpolated_path)
        
        # Create cropped images in /temp_chunks/
        rows, cols = create_chunks(interpolated_path, self.temp_dir, 
                                   x_chunk_size=int(self.chunk_size/scan_xres),
                                   y_chunk_size=int(self.chunk_size/scan_yres))
        ignore_images = [cropped_path, interpolated_path, geotiff_path]

        input_files = glob.glob(os.path.join(self.temp_dir, "*"))

        # Copy metadata to model output
        for i, input_file_path in enumerate(input_files):
            if input_file_path in ignore_images:
                continue

            output_tiff_file_path = test([input_file_path], WEIGHTS_PATH,
                                        x_chunk_size=int(self.chunk_size/scan_xres),
                                        y_chunk_size=int(self.chunk_size/scan_yres))[0][0]

            
            copy_tiff_metadata(input_file_path, output_tiff_file_path)

            set_progress(int(100.0*i/len(input_files)))


        # Merge the chunks
        merge_chunks(self.temp_dir, rows, cols, output_path, save_model_output)
        crop_tiff(output_path, output_path, width, height)

        merge_transparent_parts(cropped_path, output_path, output_path)
        copy_tiff_metadata(cropped_path, output_path)

def printProgress(value: int):
    """ Replacement for the GUI progress bar """

    print('progress: {} %'.format(value))
