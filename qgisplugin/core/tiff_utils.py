import os
import math
import numpy as np
from PIL import Image
import cv2

from osgeo import gdal, gdalconst


def crop_tiff(input_tiff, output_tiff, width, height, start_x=0, start_y=0):
    """
    Crop a TIFF image to a specific width and height.
    """
    # Open the input TIFF file
    dataset = gdal.Open(input_tiff)
    if dataset is None:
        raise ValueError("Unable to open input TIFF file.")
    
    num_bands = dataset.RasterCount

    
    # Get the geo-transform and projection
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    
    # Create a new dataset for the cropped image
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_tiff, width, height, num_bands, dataset.GetRasterBand(1).DataType)
    
    # Set the geo-transform and projection
    new_geotransform = list(geotransform)
    new_geotransform[0] += start_x * geotransform[1] + start_y * geotransform[2]
    new_geotransform[3] += start_x * geotransform[4] + start_y * geotransform[5]
    out_dataset.SetGeoTransform(new_geotransform)
    out_dataset.SetProjection(projection)

    # Read and write data for each band
    for band_num in range(1, num_bands + 1):
        # Read the specified window
        band = dataset.GetRasterBand(band_num)
        out_band = out_dataset.GetRasterBand(band_num)

        cropped_array = band.ReadAsArray(start_x, start_y, width, height)

        out_band.WriteArray(cropped_array)
    
    # Flush and close the datasets
    out_band.FlushCache()
    out_dataset.FlushCache()
    del dataset
    del out_dataset

def create_chunks(input_path, output_dir, chunk_size=501):
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

            extended_width = math.ceil(width/chunk_size) * chunk_size
            extended_height = math.ceil(height/chunk_size) * chunk_size

            # Create chunk file name
            chunk_filename = f"chunk_{i}_{j}.tif"
            chunk_path = os.path.join(output_dir, chunk_filename)

            # Create chunk dataset
            driver = gdal.GetDriverByName("GTiff")
            chunk_dataset = driver.Create(chunk_path, extended_width, extended_height, num_bands, dataset.GetRasterBand(1).DataType)

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

                # PAD the data
                new_data = np.zeros((501, 501))
                new_data[:len(data), :len(data[0])] = data

                chunk_band.WriteArray(new_data)

            # Close chunk dataset
            chunk_dataset = None

    # Close original dataset
    dataset = None

    return y_chunks, x_chunks # rows, cols

def merge_chunks(output_dir, rows, cols, output_path, save_model_output):
    # Merge the tiff files
    chunk_tiff_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if "_pred.tiff" in f]
    gdal_merge_cmd = f"gdal_merge.py -o {output_path} " + " ".join(chunk_tiff_files)
    os.system(gdal_merge_cmd)

    if save_model_output:
        # Merge the npy files
        chunk_npy_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if "_pred.npy" in f]
        chunk_shape = np.load(chunk_npy_files[0]).shape
        final_image_shape = (chunk_shape[0], chunk_shape[1], rows*chunk_shape[2], cols*chunk_shape[3])  # rows, cols
        final_image = np.zeros(final_image_shape)

        for r in range(rows):
            for c in range(cols):
                chunk_file_name = [filename for filename in chunk_npy_files if f'{c}_{r}_pred.npy' in filename][0]
                chunk_arr = np.load(chunk_file_name)
                
                start_row = chunk_shape[2]*r
                start_col = chunk_shape[3]*c
                final_image[:, :, start_row:start_row+chunk_shape[2], start_col:start_col+chunk_shape[3]] = chunk_arr
        
        npy_output_path = os.path.splitext(output_path)[0] + ".npy"
        np.save(npy_output_path, final_image)

def get_tiff_size(tiff_path):
    dataset = gdal.Open(tiff_path)
    if dataset is None:
        raise ValueError("Unable to open TIFF file.")
    
    # Get the width and height
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    
    # Close the dataset
    del dataset
    
    return width, height

def merge_transparent_parts(image1_path, image2_path, output_path):
    """
    Merge the transparent parts of image1 into image2.
    """
    # Open both images
    image1 = Image.open(image1_path).convert("RGBA")
    image2 = Image.open(image2_path).convert("RGBA")

    # Ensure image2 is the same size as image1
    if image1.size != image2.size:
        raise ValueError("Both images must be the same size")

    # Split image1 into its components
    r1, g1, b1, a1 = image1.split()
    # Split image2 into its components
    r2, g2, b2, a2 = image2.split()

    # Create a mask where image1 is transparent
    transparency_mask = Image.eval(a1, lambda alpha: 255 if alpha == 0 else 0)

    # Paste the transparent parts of image1 into image2 using the mask
    image2.paste(image1, (0, 0), transparency_mask)

    # Save the resulting image
    image2.save(output_path, format='TIFF')

def linear_interpolate_transparent(input_tiff_path, output_tiff_path):
    dataset = gdal.Open(input_tiff_path, gdal.GA_ReadOnly)
    
    # Read the image bands
    bands = [dataset.GetRasterBand(i+1).ReadAsArray() for i in range(dataset.RasterCount)]

    
    # Separate the alpha channel (assuming it is the last band)
    if len(bands) == 4:
        alpha_channel = bands[-1]
        image = np.dstack(bands[:-1])
        
        # Create a mask where alpha is 0
        mask = (alpha_channel == 0)
        
        # Fill the mask with nearest neighboring pixels
        inpainted_image = cv2.inpaint(image, mask.astype(np.uint8), 3, cv2.INPAINT_NS)
        alpha_channel[mask] = 255
        
        # Combine the inpainted image with the alpha channel
        result_image = np.dstack((*cv2.split(inpainted_image), alpha_channel))
    else:
        result_image = np.dstack(bands)
    
    # Save the result as a new GeoTIFF
    driver = gdal.GetDriverByName("GTiff")
    out_dataset = driver.Create(output_tiff_path, dataset.RasterXSize, dataset.RasterYSize, result_image.shape[2], gdal.GDT_Byte)
    
    for i in range(result_image.shape[2]):
        out_band = out_dataset.GetRasterBand(i+1)
        out_band.WriteArray(result_image[:,:,i])
    
    # Copy georeference info
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())
    
    out_dataset.FlushCache()
    del out_dataset

def copy_tiff_metadata(input_file_path, output_file_path):
    '''Copies GeoTiff metadata from input_file_path to output_file_path'''

    tif_with_RPCs = gdal.Open(input_file_path, gdalconst.GA_ReadOnly)
    tif_without_RPCs = gdal.Open(output_file_path,gdalconst.GA_Update)
    
    geo_trans = tif_with_RPCs.GetGeoTransform()
    tif_without_RPCs.SetGeoTransform(geo_trans)
    tif_without_RPCs.SetProjection(tif_with_RPCs.GetProjection())

    rpcs = tif_with_RPCs.GetMetadata('RPC')
    tif_without_RPCs.SetMetadata(rpcs ,'RPC')

    del(tif_with_RPCs)
    del(tif_without_RPCs)