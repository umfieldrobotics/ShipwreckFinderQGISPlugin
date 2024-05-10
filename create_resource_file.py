# -*- coding: utf-8 -*-
"""
| ----------------------------------------------------------------------------------------------------------------------
| Date                : February 2020                                todo: change date, copyright and email in all files
| Copyright           : © 2020 by Benjamin Jakimow and Ann Crabbé
| Email               : acrabbe.foss@gmail.com
| Acknowledgements    : Based on 'Create A QGIS Plugin' [https://bitbucket.org/kul-reseco/create-qgis-plugin]
|                       Crabbé Ann, Jakimow Benjamin, Somers Ben; funded by BELSPO STEREO III (LUMOS - SR/01/321)
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
import pathlib
import PyQt5.pyrcc_main as py_rcc

# user variables todo: adapt these if you made changes to the folder or file names
QGIS_PLUGIN_FOLDER = 'qgisplugin'
IMAGE_FOLDER = 'images'
RESOURCES_FILE = 'cqp_resources.qrc'


def compile_plugin_resources():
    image_folder = pathlib.Path(__file__).resolve().parents[0] / QGIS_PLUGIN_FOLDER / IMAGE_FOLDER
    original = RESOURCES_FILE
    output = '{}_rc.py'.format(os.path.splitext(original)[0])

    # save settings
    last_level = py_rcc.compressLevel
    last_threshold = py_rcc.compressThreshold
    last_cwd = os.getcwd()

    # increase compression level and move to *.qrc's directory
    py_rcc.compressLevel = 7
    py_rcc.compressThreshold = 100
    os.chdir(image_folder.as_posix())

    assert py_rcc.processResourceFile([original], output, False)

    # restore previous settings
    py_rcc.compressLevel = last_level
    py_rcc.compressThreshold = last_threshold
    os.chdir(last_cwd)

    print("Recourse file '{}' created.".format(image_folder / output))


if __name__ == "__main__":
    compile_plugin_resources()
