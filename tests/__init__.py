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
import os
import numpy as np
import unittest
from qgis.core import QgsApplication
app = QgsApplication([], True)
app.initQgis()


class ExtendedUnitTesting(unittest.TestCase):

    @staticmethod
    def clean_up(clean_up_files):
        for file in clean_up_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except PermissionError:
                    pass

    def assertEqualFloatArray(self, array1, array2, decimals):
        if array1.ndim == 1:
            pix = array1.shape[0]
        elif array1.ndim == 2:
            pix = array1.shape[0] * array1.shape[1]
        elif array1.ndim == 3:
            pix = array1.shape[0] * array1.shape[1] * array1.shape[2]
        else:
            assert 0

        array1_line = np.reshape(array1, pix)
        array2_line = np.reshape(array2, pix)
        for i in range(pix):
            self.assertAlmostEqual(array1_line[i], array2_line[i], places=decimals)

    def assertEqualIntArray(self, array1, array2):
        if array1.ndim == 1:
            pix = array1.shape[0]
        elif array1.ndim == 2:
            pix = array1.shape[0] * array1.shape[1]
        elif array1.ndim == 3:
            pix = array1.shape[0] * array1.shape[1] * array1.shape[2]
        else:
            assert 0

        array1_line = np.reshape(array1, pix)
        array2_line = np.reshape(array2, pix)
        for i in range(pix):
            self.assertEqual(array1_line[i], array2_line[i])

    def assertEqualStringArray(self, array1, array2):
        x = array1.shape[0]
        for i in range(x):
            self.assertEqual(array1[i], array2[i])
