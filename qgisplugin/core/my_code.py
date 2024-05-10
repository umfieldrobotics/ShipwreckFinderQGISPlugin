# -*- coding: utf-8 -*-
"""
| ----------------------------------------------------------------------------------------------------------------------
| Date                : October 2020                                 todo: change date, copyright and email in all files
| Copyright           : © 2020 by Ann Crabbé
| Email               : acrabbe.foss@gmail.com
| Acknowledgements    : Based on 'Create A QGIS Plugin' [https://bitbucket.org/kul-reseco/create-qgis-plugin]
|                       Crabbé Ann and Somers Ben; funded by BELSPO STEREO III (Project LUMOS - SR/01/321)
|
| This file is part of the [INSERT PLUGIN NAME] plugin and [INSERT PYTHON PACKAGE NAME] python package.
| todo: make sure to fill out your plugin name and python package name here.
|
| This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
| License as published by the Free Software Foundation, either version 3 of the License, or any later version.
|
| This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
| warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
|
| You should have received a copy of the GNU General Public License (COPYING.txt). If not see www.gnu.org/licenses.
| ----------------------------------------------------------------------------------------------------------------------
"""
import numpy as np


class MyCode:
    """
    todo: Give a short summary of what this code does here.

    e.g.: This script is a super simple example of a set of functions: one to multiply an image with a factor, one
    to add a constant to an entire image and one to set all values below a threshold to 0.

    This script only contains the mathematical part of your code and should be completely independent of i/o.
    You start from matrices, integers and other variables, and not from files or widgets!

    """

    def __init__(self, image: np.ndarray, normalize: bool = False, quotient: int = 255):
        """
        todo: describe your variables

        :param image: e.g. "Input image [n bands x m rows x b bands]"
        :param normalize: e.g. "Set to true to normalize the image"
        :param quotient: e.g. "Normalisation quotient for the image. Ignored if variable_2 is set to False
        """
        self.image = image
        if normalize:
            self.image = self.image / quotient

        # variables required for using the algorithm inside a UI
        self.set_progress = None
        self.print_log = None

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

        self.set_progress = set_progress if set_progress else printProgress
        self.print_log = log if log else print

        # step 1: add 0.01 to the image
        self.add_to_image(constant)
        self.print_log('Added {} to the image'.format(constant))

        self.set_progress(30)

        # step 2: get the indices of all pixels that are below the threshold
        indices_to_set_to_zero = np.where(self.image < threshold)
        self.set_progress(60)

        # step 3: set those pixels to 0
        new_image = np.copy(self.image)
        new_image[indices_to_set_to_zero] = 0
        self.set_progress(90)

        self.print_log('Core processing done.')

        return new_image


def printProgress(value: int):
    """ Replacement for the GUI progress bar """

    print('progress: {} %'.format(value))
