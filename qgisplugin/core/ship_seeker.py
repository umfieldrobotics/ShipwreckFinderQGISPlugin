# -*- coding: utf-8 -*-

import numpy as np
import torch

OUTPUT_PATH = "/home/frog/dev/output/out.tif"
LAYER_NAME = "Mesa"

class ShipSeeker:
    """
    The code responsible for splitting up an input raster image into chunks, converting each to 
    ML format, finding potential shipwrecks using pytorch, then combining the resulting chunks to
    a final raster image. 
    """

    #TODO: do this
    def __init__(self, raster_layer, normalize: bool = False, quotient: int = 255):
        """
        todo: describe your variables

        :param image: e.g. "Input image [n bands x m rows x b bands]"
        :param normalize: e.g. "Set to true to normalize the image"
        :param quotient: e.g. "Normalisation quotient for the image. Ignored if variable_2 is set to False
        """
        self.raster_layer = raster_layer

    def add_to_image(self, constant: float) -> np.ndarray:
        """
        Add a constant to an image.

        :param constant: The constant to add to each pixel of the image.
        :return: The new image.
        """

        return self.image + constant

    def execute(self, constant: float, threshold: float, set_progress: callable = None,
                log: callable = print) -> np.ndarray:
        """
        This part is usually the core of your plugin: this function is called when the user clicks "run".

        Here we don't do anything special: we add a number to an image and then set all values in an image to 0 where
        they are below a given threshold.

        :param constant: The constant to add to each pixel of the image.
        :param threshold: all values below this threshold are set to 0
        :param set_progress: communicate progress (refer to the progress bar in case of GUI; otherwise print to console)
        :param log: communicate messages (refer to the print_log tab in the GUI; otherwise print to the console)
        :return: the new image
        """

        

        # self.set_progress = set_progress if set_progress else printProgress
        # self.print_log = log if log else print

        # # step 1: add 0.01 to the image
        # self.add_to_image(constant)
        # self.print_log('Added {} to the image'.format(constant))

        # self.set_progress(30)

        # # step 2: get the indices of all pixels that are below the threshold
        # indices_to_set_to_zero = np.where(self.image < threshold)
        # self.set_progress(60)

        # # step 3: set those pixels to 0
        # new_image = np.copy(self.image)
        # new_image[indices_to_set_to_zero] = 0
        # self.set_progress(90)

        self.print_log('Core processing done.')

        return new_image


def printProgress(value: int):
    """ Replacement for the GUI progress bar """

    print('progress: {} %'.format(value))
