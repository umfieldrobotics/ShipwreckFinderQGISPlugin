# -*- coding: utf-8 -*-

from ..safe_libs_setup import setup_libs, safe_import_ml_libraries

setup_libs()
libs = safe_import_ml_libraries()


import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

class BoundingBoxExtractor:
    '''
    Extracts bounding boxes for the image in world coordinates 
    or pixel coordinates. Operates on RGB/RGBA images.
    '''
    def __init__(self, segmentation_image, geotransform): 
        
        self.segmentation_image = segmentation_image
        self.geotransform = geotransform

        # Make conversion between pixel area and metric area
        pixel_width = abs(self.geotransform[1])   # Horizontal size of a pixel in meters
        pixel_height = abs(self.geotransform[5])  # Vertical size of a pixel in meters
        self.pixel_area_meters = pixel_width * pixel_height

        # Get unique values in segmentation image and use them as masks...
        pixels = segmentation_image.reshape(-1, 4)
        self.unique_rgba, _ = np.unique(pixels, axis=0, return_inverse=True)
        
        self.masks = {}
        for i, rgba in enumerate(self.unique_rgba):
            mask = (pixels == rgba).all(axis=1).reshape(segmentation_image.shape[:2])
            self.masks[tuple(rgba)] = mask.astype(np.uint8) 

    def get_unique_colors(self):
        '''Gets the unique rgba colors present in the image'''
        return self.unique_rgba
    
    def get_masks(self):
        '''Gets a mask for each pixel color in the image'''
        return self.masks
    
    def get_mask(self,color):
        return self.masks[tuple(color)]
        

    def extract_pixel_bb(self, segmentation_mask, min_area_threshold_pixels=0, max_area_threshold_pixels=10000000):
        # Find external contours from the segmentation mask
        cnts = cv2.findContours(segmentation_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        extracted_bounding_boxes = [] 

        if len(cnts)>0:
            for (i, c) in enumerate(cnts):
                # Calculate the area of the contour and skip small or very large contours outside the thresholds
                if cv2.contourArea(c) < min_area_threshold_pixels or cv2.contourArea(c) > max_area_threshold_pixels:
                    continue
                # Get the minimum-area rotated rectangle enclosing the contour
                box = cv2.minAreaRect(c)
                # Convert box to four corner points
                box = cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                # Order points consistently
                rect = self.__order_points(box)
                # Save the bounding box
                extracted_bounding_boxes.append(rect)

        return extracted_bounding_boxes
    
    def extract_metric_bb(self, segmentation_mask, min_area_threshold_meters=0, max_area_threshold_meters=10000000):
        # Convert area thresholds from meters to pixel units using the known pixel area
        min_area_threshold_pixels = min_area_threshold_meters / self.pixel_area_meters
        max_area_threshold_pixels = max_area_threshold_meters / self.pixel_area_meters

        # Extract bounding boxes in pixel space
        pixel_bb = self.extract_pixel_bb(segmentation_mask, min_area_threshold_pixels, max_area_threshold_pixels)

        map_bb = []
        for i, pixel_coords in enumerate(pixel_bb):       
                # Convert each pixel corner into map coordinates using geotransform
                map_coords = [self.__pixel_to_map_coordinates(x, y, self.geotransform) for (x, y) in pixel_coords]
                map_bb.append(map_coords)

        return map_bb
        
    def __order_points(self, pts):
        # Initialize container for the ordered coordinates
        rect = np.zeros((4, 2), dtype="float32")
        rect = [0]*4

        # Top-left point will have smallest sum, bottom-right will have largest
        s = pts.sum(axis=1)
        rect[0] = tuple(pts[np.argmin(s)])
        rect[2] = tuple(pts[np.argmax(s)])

        # Top-right point will have smallest difference, bottom-left will have largest
        diff = np.diff(pts, axis=1)
        rect[1] = tuple(pts[np.argmin(diff)])
        rect[3] = tuple(pts[np.argmax(diff)])

        return rect
    
    def __pixel_to_map_coordinates(self, x, y, geotransform):
        """
        Convert pixel coordinates to map coordinates using the geotransform.
        
        :param x: Pixel x-coordinate
        :param y: Pixel y-coordinate
        :param geotransform: GDAL geotransform
        :return: Tuple (map_x, map_y) - Map coordinates
        """
        
        map_x = geotransform[0] + x * geotransform[1] + y * geotransform[2]
        map_y = geotransform[3] + x * geotransform[4] + y * geotransform[5]
        return map_x, map_y
    
