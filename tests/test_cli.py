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
from os.path import join, dirname, abspath
from qgisplugin.interfaces.my_cli import create_parser, run_code
from qgisplugin.interfaces import import_image
from tests import ExtendedUnitTesting

DATA_FOLDER = join(dirname(abspath(__file__)), "data")


class TestMyCLI(ExtendedUnitTesting):

    def test_my_cli(self):
        parser = create_parser()
        args = [join(DATA_FOLDER, 'image.tif'), '-n=255', '-o={}'.format(join(DATA_FOLDER, 'output.tif'))]
        output_path = run_code(parser.parse_args(args))

        result, _ = import_image(output_path)
        self.assertEqual(result.shape[0], 4)
        self.assertEqual(result.shape[1], 1201)
        self.assertEqual(result.shape[2], 1301)

        # clean up
        self.clean_up([output_path])

    def test_system_exit(self):
        # code throws error when using empty parser
        parser = create_parser()
        with self.assertRaises(SystemExit):
            run_code(parser.parse_args([]))
