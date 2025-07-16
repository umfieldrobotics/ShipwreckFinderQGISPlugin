import os
import math
import numpy as np
from PIL import Image
import cv2
import rasterio

from osgeo import gdal, gdalconst
from qgisplugin.core.data import normalize_nonzero


def crop_tiff(input_tiff, output_tiff, width, height, start_x=0, start_y=0):
    """
    Crop a TIFF image to a specific width and height.
    """
    # Open the input TIFF file
    dataset = gdal.Open(input_tiff)
    # print("Copy1")
    if dataset is None:
        raise ValueError("Unable to open input TIFF file.")
    
    num_bands = dataset.RasterCount
    # print("Copy2")
    
    # Get the geo-transform and projection
    geotransform = dataset.GetGeoTransform()
    # print("Copy3")
    projection = dataset.GetProjection()
    # print("Copy4")
    
    # Create a new dataset for the cropped image
    driver = gdal.GetDriverByName('GTiff')
    # print("Copy5")
    out_dataset = driver.Create(output_tiff, width, height, num_bands, dataset.GetRasterBand(1).DataType)
    # print("Copy6")
    
    # Set the geo-transform and projection
    new_geotransform = list(geotransform)
    # print("Copy7")
    new_geotransform[0] += start_x * geotransform[1] + start_y * geotransform[2]
    # print("Copy8")
    new_geotransform[3] += start_x * geotransform[4] + start_y * geotransform[5]
    # print("Copy9")
    out_dataset.SetGeoTransform(new_geotransform)
    # print("Copy9")
    out_dataset.SetProjection(projection)
    # print("Copy10")

    # Read and write data for each band
    for band_num in range(1, num_bands + 1):
        # Read the specified window
        band = dataset.GetRasterBand(band_num)
        out_band = out_dataset.GetRasterBand(band_num)

        cropped_array = band.ReadAsArray(start_x, start_y, width, height)

        out_band.WriteArray(cropped_array)
    # print("Copy11")
    # Flush and close the datasets
    out_band.FlushCache()
    # print("Copy12")
    out_dataset.FlushCache()
    # print("Copy13")
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
                # print(f"Saving chunk band {band_num}")
                band = dataset.GetRasterBand(band_num)
                chunk_band = chunk_dataset.GetRasterBand(band_num)

                data = band.ReadAsArray(x_offset, y_offset, width, height)

                # PAD the data
                # new_data = np.zeros((501,501))
                # new_data[:len(data), :len(data[0])] = data

                chunk_band.WriteArray(data)

            # Close chunk dataset
            chunk_dataset = None

    # Close original dataset
    dataset = None

    return y_chunks, x_chunks # rows, cols

def merge_chunks(output_dir, rows, cols, output_path, save_model_output):
    # Merge the tiff files
    chunk_tiff_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if "_pred.tiff" in f]

    # print(f"Chunk tiff files from merge_chunks:")
    # for elt in chunk_tiff_files:
    #     print(elt)

    gdal_merge_cmd = f"gdal_merge.py -o {output_path} " + " ".join(chunk_tiff_files)
    # gdal_merge_cmd = gdal_merge_cmd + " > /home/smitd/Desktop/merge_log.txt 2>&1"
    # with open("/home/smitd/Desktop/gdal_cmd.txt", 'w') as f:
    #     f.write(gdal_merge_cmd)

    # Things to try:
    #   BIGTIFF in case it's too large for a tiff when saving (--co BIGTIFF=YES) and (--co TILED=YES)
        # --co \"BIGTIFF=YES\" --co \"TILED=YES\"
    #   Use gdalwarp in case it's memory limitations
    #   Increase file descriptor limit (1024), using (ulimit -n 8192 <or higher>)

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

    with rasterio.open(image1_path) as src1, rasterio.open(image2_path) as src2:
        # Check size compatibility
        if src1.width != src2.width or src1.height != src2.height:
            raise ValueError("Both images must be the same size")

        # print(f"Image1 bands: {src1.count}, Image2 bands: {src2.count}")

        # Read both images
        image1 = src1.read()
        image2 = src2.read()

        # image1[0] = normalize_nonzero(image1[0])

        # print("Merge2")

        if src1.count == 2:
            # Use alpha channel
            alpha1 = image1[1]
            transparent_mask = (alpha1 == 0)
        elif src1.count == 1:
            # Assume fully opaque: no transparency
            transparent_mask = np.zeros_like(image1[0], dtype=bool)
        else:
            raise ValueError("Image1 must have 1 or 2 bands")

        # print("Merge3")
        # Replace pixels in image2 where image1 is transparent
        for b in range(4):  # For each band in image2 (RGBA)
            if b < 3:
                image2[b][transparent_mask] = image1[0][transparent_mask]
            else:
                image2[b][transparent_mask] = 0  # Set alpha to 0

        # print("Merge4")
        # Write the result to a new file
        profile = src2.profile
        profile.update({
            "count": 4,
            "dtype": image2.dtype
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(image2)



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

    # print("Meta1")

    tif_with_RPCs = gdal.Open(input_file_path, gdalconst.GA_ReadOnly)
    # print("Meta2")
    tif_without_RPCs = gdal.Open(output_file_path,gdalconst.GA_Update)
    # print("Meta3")
    
    geo_trans = tif_with_RPCs.GetGeoTransform()
    # print("Meta4")
    tif_without_RPCs.SetGeoTransform(geo_trans)
    # print("Meta5")
    tif_without_RPCs.SetProjection(tif_with_RPCs.GetProjection())
    # print("Meta6")

    # print("THIS IS THE PROJECTION:" , tif_with_RPCs.GetProjection())
    # print("Meta7")

    rpcs = tif_with_RPCs.GetMetadata('RPC')
    # print("Meta8")
    tif_without_RPCs.SetMetadata(rpcs ,'RPC')
    # print("Meta9")

    del(tif_with_RPCs)
    del(tif_without_RPCs)


def remove_invalid_pixels(cropped_path, input_path, output_path):
    with rasterio.open(cropped_path) as crp, rasterio.open(input_path) as inp:
        cropped = crp.read(1)
        data = inp.read()
        # NO_DATA = -9999
        # print(f"Cropped dim: {cropped.shape}, Input dims: {input.shape}")
        # print(f"Cropped corner val: {cropped[0, 0]}, Cropped center val: {cropped[500, 700]}")
        # cv2.imwrite("/home/smitd/Documents/Copied_Temp_Chunks/cropped_out.png", cropped)
        # print(f"Cropped band: {crp.count}, Input band: {inp.count}")

        invalid_mask = ((cropped >= 1000) | (cropped <= -1000)).astype(np.uint8)
        
        # invalid_mask_3d = invalid_mask[np.newaxis, :, :] # shape (1, H, W)
        # invalid_mask_3d = np.repeat(invalid_mask_3d, data.shape[0], axis=0)

        # data[invalid_mask_3d == 1] = NO_DATA
        data[0][invalid_mask == 1] = 0
        data[1][invalid_mask == 1] = 0
        data[2][invalid_mask == 1] = 127
        data[3][:,:] = 255

        # nodata_mask = np.any(data[3] == NO_DATA, axis=0) # shape (H, W)
        # data[3] = np.where(nodata_mask, 0, 255).astype(np.uint8)

        profile = inp.profile
        profile.update(
            dtype=rasterio.uint8,
            count=data.shape[0]
        )

        data = data.astype(np.uint8)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data.astype(np.float32))


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

