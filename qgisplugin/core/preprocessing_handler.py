# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

from osgeo import gdal

class PreprocessingHandler:
    def __init__(self): 
        pass

    def normalize(self, depth_array, set_progress: callable = None, 
                  kernel_size=200, inpaint_radius=10):
        # Handle both NoData values (NOAA BAG == 1000000) (Field XYZ == -9999)
        depth_array[depth_array >= 10000] = np.nan
        depth_array[depth_array <= -9999] = np.nan

        # print(f"Kernel Size: {kernel_size}")
        # print(f"Depth Array Size: {depth_array.shape}")
        H, W = depth_array.shape

        if (H > 5000 and W > 5000) or (H * W > 25000000):
            final_rgb_out = np.zeros((H, W, 3), dtype=np.uint8)
            overlap = 128
            stride = 1024 - (2 * overlap)

            global_min = np.inf
            global_max = -np.inf

            for i in range(0, H, stride):
                for j in range(0, W, stride):
                    i_end = min(i + stride, H)
                    j_end = min(j + stride, W)

                    arr = depth_array[i:i_end, j:j_end]

                    min_val, max_val = self.get_min_max(arr, kernel_size)
                    if min_val < global_min:
                        global_min = min_val
                    if max_val > global_max:
                        global_max = max_val

                if callable(set_progress):
                    # 20 has already been set by the time we get here, cap this function at 30
                    update = int((i * 10) // H)
                    set_progress(20 + update)

            print(f"Min, Max : {min_val}, {max_val}")

            # print(f"Total Depth Array Size: {H}, {W}")
            for i in range(0, H, stride):
                for j in range(0, W, stride):
                    i_start = max(i - overlap, 0)
                    j_start = max(j - overlap, 0)
                    i_end = min(i + stride + overlap, H)
                    j_end = min(j + stride + overlap, W)
                    # i_end = min(i + stride, H)
                    # j_end = min(j + stride, W)
                    print(f"Processing rows {i_start}:{i_end} and cols {j_start}:{j_end}")

                    depth_array_piece = depth_array[i_start:i_end, j_start:j_end]

                    out_i = min(i + stride, H)
                    out_j = min(j + stride, W)
                    ci_start = overlap if i > 0 else 0
                    cj_start = overlap if j > 0 else 0
                    ci_end = ci_start + (out_i - i)
                    cj_end = cj_start + (out_j - j)

                    final_rgb_out[i:out_i, j:out_j] = self.normalize_helper(depth_array_piece, None, kernel_size, inpaint_radius, global_min, global_max)[ci_start:ci_end, cj_start:cj_end]
                
                if callable(set_progress):
                    # 20 has already been set by the time we get here, cap this function at 80
                    update = int((i * 50) // H)
                    set_progress(30 + update)

            return final_rgb_out
        
        else:

            return self.normalize_helper(depth_array, set_progress, kernel_size, inpaint_radius)
        
    def normalize_helper(self, depth_array, set_progress: callable = None, 
                  kernel_size=200, inpaint_radius=10, min_val=None, max_val=None):
        
        nan_mask = np.isnan(depth_array).astype(np.float32)
        depth_array_with_zeros = np.where(nan_mask, 0, depth_array).astype(np.float32)

        # Kernel
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

        matrix_sum = cv2.filter2D(depth_array_with_zeros, -1, kernel)
        not_nan_count = cv2.filter2D(1-nan_mask, -1, kernel)

        if callable(set_progress):
            set_progress(30)

        local_mean_img = matrix_sum / np.maximum(not_nan_count, 1.0)

        local_mean_img[nan_mask == 1] = 0 # TODO: is this right?

        # Sbtraced image (x_i - u)
        subtracted_img = np.abs(depth_array_with_zeros - local_mean_img)

        # Compute the std_dev
        subtracted_img_sqr = subtracted_img ** 2

        subtracted_img_sqr_summed = cv2.filter2D(subtracted_img_sqr, -1, kernel)
        subtracted_img_sqr_summed[subtracted_img_sqr_summed < 0] = 0

        if callable(set_progress):
            set_progress(40)

        local_variance_img = subtracted_img_sqr_summed / np.maximum(not_nan_count, 1.0)

        local_stddev_img = np.sqrt(local_variance_img)

        local_stddev_img[local_stddev_img == 0] = 1 # make sure re we're not dividing by 0


        # DO a cutoff instead
        # subtracted_img[np.abs(subtracted_img)>local_stddev_img] *= 10

        # print("Depth min/max:", np.nanmin(depth_array), np.nanmax(depth_array))
        # print("Non-NaN pixels:", np.count_nonzero(~np.isnan(depth_array)))

        normalized_raw = subtracted_img / np.sqrt(local_stddev_img)

        if callable(set_progress):
            set_progress(50)

            normalized_img = cv2.normalize(normalized_raw.astype(np.float32), None, .5, 1.0, cv2.NORM_MINMAX)
            normalized_img[nan_mask == 1] = 0

            # normalized_img = normalized_raw.copy()

            # print("Normalized raw min/max:", normalized_raw.min(), normalized_raw.max())
            # print("After normalize() min/max:", normalized_img.min(), normalized_img.max())

            inpainted = cv2.inpaint((normalized_img * 255).astype(np.uint8), nan_mask.astype(np.uint8), inpaint_radius, cv2.INPAINT_TELEA)
            rgb_grayscale_image = cv2.cvtColor((inpainted).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            
            set_progress(80)

            return rgb_grayscale_image
        else:
            return self.manual_inpaint_norm_minmax(normalized_raw, 0.5, 1.0, min_val, max_val, nan_mask, inpaint_radius)
    
    def get_min_max(self, depth_array, kernel_size=200):
        nan_mask = np.isnan(depth_array).astype(np.float32)
        depth_array_with_zeros = np.where(nan_mask, 0, depth_array).astype(np.float32)
        # Kernel

        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        matrix_sum = cv2.filter2D(depth_array_with_zeros, -1, kernel)
        not_nan_count = cv2.filter2D(1-nan_mask, -1, kernel)
        local_mean_img = matrix_sum / np.maximum(not_nan_count, 1.0)
        local_mean_img[nan_mask == 1] = 0 # TODO: is this right?

        # Sbtraced image (x_i - u)
        subtracted_img = np.abs(depth_array_with_zeros - local_mean_img)

        # Compute the std_dev
        subtracted_img_sqr = subtracted_img ** 2
        subtracted_img_sqr_summed = cv2.filter2D(subtracted_img_sqr, -1, kernel)
        subtracted_img_sqr_summed[subtracted_img_sqr_summed < 0] = 0
        local_variance_img = subtracted_img_sqr_summed / np.maximum(not_nan_count, 1.0)
        local_stddev_img = np.sqrt(local_variance_img)
        local_stddev_img[local_stddev_img == 0] = 1 # make sure re we're not dividing by 0
        # Normalize
        normalized_raw = subtracted_img / np.sqrt(local_stddev_img)
        # Return min and max values
        return np.min(normalized_raw), np.max(normalized_raw)


    def manual_inpaint_norm_minmax(self, arr, alpha, beta, src_min, src_max, nan_mask, inpaint_radius):
        '''
            Performs the equivalent of cv2.normalize, and then applies inpainting and grayscales to RGB
        '''
        # src_min = np.min(src)
        # src_max = np.max(src)

        # print(f"Nan mask sum: {np.sum(nan_mask)}")
        
        # Handle edge case where all values are the same
        if src_max == src_min:
            return np.full(arr.shape, alpha, dtype=arr.dtype)
        
        # Scale and shift: new_val = (val - src_min) / (src_max - src_min) * (beta - alpha) + alpha
        scale = (beta - alpha) / (src_max - src_min)
        out = (arr - src_min) * scale + alpha

        out[nan_mask == 1] = 0

        inpainted = cv2.inpaint((out * 255).astype(np.uint8), nan_mask.astype(np.uint8), inpaint_radius, cv2.INPAINT_TELEA)
        rgb_grayscale_image = cv2.cvtColor((inpainted).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        return rgb_grayscale_image
    
    