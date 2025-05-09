# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

class PreprocessingHandler:
    def __init__(self): 
        pass

    # def normalize(self, depth_array, set_progress: callable = None, 
    #               kernel_size=200, inpaint_radius=10, x_res=1.0, y_res=1.0):
    #     depth_array[depth_array > 10000] = np.nan


    #     nan_mask = np.isnan(depth_array).astype(np.float32)
    #     depth_array_with_zeros = np.where(nan_mask, 0, depth_array).astype(np.float32)

    #     # Kernel
    #     kernel = np.ones((kernel_size/x_res, kernel_size/y_res), dtype=np.float32)

    #     # Find totals across all of depth array
    #     matrix_sum = cv2.filter2D(depth_array_with_zeros, -1, kernel)
    #     not_nan_count = cv2.filter2D(1-nan_mask, -1, kernel) 

    #     set_progress(30)

    #     local_mean_img = matrix_sum / np.maximum(not_nan_count, 1.0)

    #     local_mean_img[nan_mask == 1] = 0 # TODO: is this right?



    #     # Sbtraced image (x_i - u)
    #     subtracted_img = np.abs(depth_array_with_zeros - local_mean_img)

    #     # Compute the std_dev
    #     subtracted_img_sqr = subtracted_img ** 2

    #     subtracted_img_sqr_summed = cv2.filter2D(subtracted_img_sqr, -1, kernel)
    #     subtracted_img_sqr_summed[subtracted_img_sqr_summed < 0] = 0
    #     set_progress(40)

    #     local_variance_img = subtracted_img_sqr_summed / np.maximum(not_nan_count, 1.0)

    #     local_stddev_img = np.sqrt(local_variance_img)

    #     local_stddev_img[local_stddev_img == 0] = 1 # make sure re we're not dividing by 0


    #     # DO a cutoff instead
    #     # subtracted_img[np.abs(subtracted_img)>local_stddev_img] *= 10

    #     normalized_raw = subtracted_img / np.sqrt(local_stddev_img)

    #     set_progress(50)

    #     normalized_img = cv2.normalize(normalized_raw.astype(np.float32), None, .5, 1.0, cv2.NORM_MINMAX)
    #     normalized_img[nan_mask == 1] = 0

    #     # Inpainting, used in original plugin. Commenting out for now
    #     # inpainted = cv2.inpaint((normalized_img * 255).astype(np.uint8), nan_mask.astype(np.uint8), inpaint_radius, cv2.INPAINT_TELEA)
    #     # rgb_grayscale_image = cv2.cvtColor((inpainted).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    #     rgb_grayscale_image = cv2.cvtColor((normalized_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    #     set_progress(80)

    #     return rgb_grayscale_image

    def normalize2(self, depth_array, set_progress: callable = None, 
                  kernel_size=200, inpaint_radius=10, x_res=1.0, y_res=1.0):
        '''
        This normalization function normalizes over the entire image, not locally. This produces images better aligned
        with the training data.
        '''
        
        depth_array[depth_array > 10000] = np.nan

        nan_mask = np.isnan(depth_array).astype(np.float32) # mask with 1s where image is NaN
        depth_array_with_zeros = np.where(nan_mask, 0, depth_array).astype(np.float32)

        image = depth_array_with_zeros

        # Normalization, not including 0's
        nonzero_mask = image != 0
        if np.any(nonzero_mask):  # Ensure there are nonzero values
            image_min = np.min(image[nonzero_mask])
            image_max = np.max(image[nonzero_mask])

            if not (image_min >= 0 and image_max <= 1):  # Only normalize if needed
                image[nonzero_mask] = (image[nonzero_mask] - image_min) / (image_max - image_min)

        # Inpainting, used in original plugin. Commenting out for now

        if inpaint_radius != 0:
            inpainted = cv2.inpaint((image * 255).astype(np.uint8), nan_mask.astype(np.uint8), inpaint_radius, cv2.INPAINT_TELEA)
            processed_image = cv2.cvtColor((inpainted).astype(np.uint8), cv2.COLOR_GRAY2RGB) # rgb_grayscale_image
        else:
            processed_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB) 
        # processed_image = np.stack((image,image,image), axis = 2)

        set_progress(80)

        return processed_image, nonzero_mask