import os
import sys
def setup_libs():
    current_dir = os.path.dirname(__file__)
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, 'libs')):
            libs_dir = os.path.join(current_dir, 'libs')
            if libs_dir not in sys.path:
                sys.path.insert(0, libs_dir)
            return
        current_dir = os.path.dirname(current_dir)
setup_libs()

import math
import numpy as np
import cv2
import rasterio
import csv

from osgeo import gdal, gdalconst, gdal_array
from qgisplugin.core.data import normalize_nonzero


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

def create_chunks(input_path, output_dir, chunk_size=400):
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

            for band_num in range(1, num_bands + 1):
                chunk_band = chunk_dataset.GetRasterBand(band_num)
                chunk_band.SetNoDataValue(-9999)

                # Fill entire chunk with NoData
                nodata_array = np.full((extended_height, extended_width), -9999, dtype=gdal_array.GDALTypeCodeToNumericTypeCode(dataset.GetRasterBand(1).DataType))
                chunk_band.WriteArray(nodata_array)


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
                
                chunk_band.WriteArray(data)


            # Close chunk dataset
            chunk_dataset = None

    # Close original dataset
    dataset = None

    return y_chunks, x_chunks # rows, cols

def merge_chunks(output_dir, rows, cols, output_path, save_model_output, is_basnet=False):
    # Merge the tiff files
    chunk_tiff_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if "_pred.tiff" in f]


    if len(chunk_tiff_files) <= 50:
        gdal_merge_cmd = f"gdal_merge.py -o {output_path} " + " ".join(chunk_tiff_files)
        os.system(gdal_merge_cmd)
    else:
        robust_gdal_merge(chunk_tiff_files, output_path)

    if save_model_output and not is_basnet:
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

def convert_png_to_tiff(input_path, output_path):
    # Open the input PNG file
    dataset = gdal.Open(input_path)
    if dataset is None:
        print("Failed to open file.")
        return

    # Define the options for the TIFF format
    options = gdal.TranslateOptions(format='GTiff')

    # Convert the image and save it as TIFF
    gdal.Translate(output_path, dataset, options=options)
    print(f"Image successfully converted to {output_path}")


def merge_transparent_parts(image1_path, image2_path, output_path):
    """
    Merge the transparent parts of image1 into image2.
    """
    from rasterio.windows import Window

    with rasterio.open(image1_path) as src1, rasterio.open(image2_path) as src2:
        if src1.width != src2.width or src1.height != src2.height:
            raise ValueError("Both images must be the same size")

        profile = src2.profile
        profile.update({
            "count": 4,
            "dtype": src2.dtypes[0]
        })

        block_size = 512

        with rasterio.open(output_path, 'w', **profile) as dst:
            for y in range(0, src1.height, block_size):
                for x in range(0, src1.width, block_size):
                    window_width = min(block_size, src1.width - x)
                    window_height = min(block_size, src1.height - y)
                    window = Window(x, y, window_width, window_height)

                    image1 = src1.read(window=window)
                    image2 = src2.read(window=window)

                    if src1.count == 2:
                        alpha1 = image1[1]
                        transparent_mask = (alpha1 == 0)
                    elif src1.count == 1:
                        transparent_mask = np.zeros((window_height, window_width), dtype=bool)
                    else:
                        raise ValueError("Image1 must have 1 or 2 bands")

                    # Merge
                    for b in range(4):
                        if b < 3:
                            image2[b][transparent_mask] = image1[0][transparent_mask]
                        else:
                            image2[b][transparent_mask] = 0

                    dst.write(image2, window=window)


def linear_interpolate_transparent(input_tiff_path, output_tiff_path):
    '''Fills in gaps (alpha=0) in input image using cv2.infill'''

    dataset = gdal.Open(input_tiff_path, gdal.GA_ReadOnly)

    original_dtype = dataset.GetRasterBand(1).DataType
    
    # Read the image bands
    bands = [dataset.GetRasterBand(i+1).ReadAsArray() for i in range(dataset.RasterCount)]

    if len(bands) == 4:
        # Separate the alpha channel 
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
    out_dataset = driver.Create(output_tiff_path, dataset.RasterXSize, dataset.RasterYSize, result_image.shape[2], original_dtype) #gdal.GDT_Byte
    
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


def ensure_valid_nodata(cropped_path, output_path):
    with rasterio.open(cropped_path) as src:
        profile = src.profile
        cropped = src.read()
        # cropped[0] is the band we care about
        crop_nodata = src.nodata

        if crop_nodata is None or crop_nodata >= 1000 or crop_nodata <= -1000:
            return
        
        nodata_mask = (cropped[0] == crop_nodata)
        cropped[0][nodata_mask] = -9999

        profile.update(dtype=rasterio.float32, nodata=-9999)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(cropped.astype(rasterio.float32))
            dst.nodata = -9999

def robust_gdal_merge(tiff_files, output_path):
    import tempfile
    import subprocess

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as list_file:
        list_path = list_file.name
        list_file.write("\n".join(tiff_files))

    # Create a temporary VRT file
    with tempfile.NamedTemporaryFile(suffix=".vrt", delete=False) as vrt_file:
        vrt_path = vrt_file.name

    try:
        # Build the VRT (virtual raster stack)
        subprocess.check_call([
            "gdalbuildvrt", "-input_file_list", list_path, vrt_path
        ])


        # Translate to a real, physical GeoTIFF file
        subprocess.check_call([
            "gdal_translate",
            "-co", "TILED=YES",
            "-co", "BLOCKXSIZE=512",
            "-co", "BLOCKYSIZE=512",
            vrt_path,
            output_path
        ])
    finally:
        for p in [list_path, vrt_path]:
            if os.path.exists(p):
                os.remove(p)


def robust_remove_invalid_pixels(cropped_path, input_path, output_path):
    invalid_pixels = 0

    with rasterio.open(cropped_path) as crp, rasterio.open(input_path) as inp:
        profile = inp.profile
        profile.update(dtype=rasterio.uint8)

        with rasterio.open(output_path, 'w', **profile) as dst:
            # Iterate through the raster in blocks
            for ji, window in inp.block_windows(1):
                cropped_block = crp.read(1, window=window)
                data_block = inp.read(window=window)

                # Mask is 1 where invalid
                invalid_mask = ((cropped_block >= 1000) | (cropped_block <= -1000))
                invalid_pixels += np.count_nonzero(invalid_mask)

                # Apply modifications
                data_block = data_block.astype(np.uint8)
                data_block[0][invalid_mask] = 0
                data_block[1][invalid_mask] = 0
                data_block[2][invalid_mask] = 127
                data_block[3][:, :] = 255

                dst.write(data_block, window=window)
    
    return invalid_pixels


def get_raster_resolution(tif_path):
    with rasterio.open(tif_path) as src:
        res_x, res_y = src.res
        return res_x, res_y


def remove_small_contours_chunked(input_path, output_path, threshold, invalid_pixels, chunk_size=2048):
    from rasterio.windows import Window
    import gc
    
    with rasterio.open(input_path) as src:
        profile = src.profile
        height, width = src.height, src.width
        
        # Calculate minimum area based on full raster dimensions
        min_area = int(((height * width) - invalid_pixels) * threshold)
        
        # Update profile for output
        profile.update({
            "count": src.count,
            "dtype": np.uint8  # Assuming output will be uint8
        })
        
        # Create output file
        with rasterio.open(output_path, 'w', **profile) as dst:
            # Process in chunks
            for row_start in range(0, height, chunk_size):
                for col_start in range(0, width, chunk_size):
                    # Calculate actual chunk dimensions (handle edge cases)
                    row_end = min(row_start + chunk_size, height)
                    col_end = min(col_start + chunk_size, width)
                    chunk_height = row_end - row_start
                    chunk_width = col_end - col_start
                    # Define window for this chunk
                    window = Window(col_start, row_start, chunk_width, chunk_height)
                    
                    # Read only this chunk
                    chunk_data = src.read(window=window)
                    
                    # Process this chunk
                    processed_chunk = process_chunk(chunk_data, min_area)
                    
                    # Write processed chunk to output
                    dst.write(processed_chunk, window=window)
                    
                    # Force garbage collection after each chunk
                    del chunk_data, processed_chunk
                    gc.collect()

def process_chunk(chunk_data, min_area):
    cleaned_data = np.copy(chunk_data)
    
    band1 = chunk_data[0].astype(np.uint8)
    band2 = chunk_data[1].astype(np.uint8) 
    band3 = chunk_data[2].astype(np.uint8)
    band4 = chunk_data[3]
    
    # Find contours in band1
    contours, _ = cv2.findContours(band1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create mask where valid contours will be drawn
    contour_mask = np.zeros_like(band1, dtype=np.uint8)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(contour_mask, [cnt], -1, 1, thickness=cv2.FILLED)
    # Apply coloring based on the mask
    cleaned_data[0] = np.where(contour_mask == 1, 127, 0) # Red in contour, 0 otherwise
    cleaned_data[1] = np.where(contour_mask == 1, 0, 0) # Green is always 0
    cleaned_data[2] = np.where(contour_mask == 1, 0, 127) # Blue in background, 0 in contour
    cleaned_data[3] = band4
    
    return cleaned_data

def generate_csv(input_path, output_csv):
    with rasterio.open(input_path) as src:
        # Extract band 1
        band1 = src.read()[0]
        contours, _ = cv2.findContours(band1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y"])

            for idx, cnt in enumerate(contours):
                # Compute moments of the contour
                m = cv2.moments(cnt)
                if m["m00"] == 0:
                    continue

                # Centroid in pixel coordinates
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])

                # Convert pixel coordinates with geospatial info
                x, y = rasterio.transform.xy(src.transform, cy, cx)

                writer.writerow([idx, x, y])




