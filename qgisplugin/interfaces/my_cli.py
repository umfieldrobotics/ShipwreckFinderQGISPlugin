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
import argparse
from os.path import abspath, join, dirname

from qgisplugin.core.my_code import MyCode
from qgisplugin.interfaces import import_image, write_image


def create_parser():
    parser = argparse.ArgumentParser(description=str(MyCode.__doc__))

    # todo: add parameters or change the names to your need
    parser.add_argument('image', type=str, help='Path to the input image.')
    parser.add_argument('-n', type=int, help='To normalize your image, set the quotient here. (default: not set)')
    parser.add_argument('-c', type=float, default=0.01, help='Add this constant to the image (default: 0.01).')
    parser.add_argument('-t', type=float, default=0.2, help='Threshold for setting cells to 0 (default: 0.2).')
    parser.add_argument('-o', type=str, help="Output file (default: in same folder with name 'output.tif'")

    return parser


def run_code(args):
    """
    Documentation: mypackage -h
    """

    # todo: read all parameters, throw errors when needed, give user feedback and run code
    image_path = abspath(args.image)
    image, metadata = import_image(image_path)

    normalize = True if args.n else False
    quotient = args.n if normalize else None
    constant = args.c
    threshold = args.t

    # run code
    result = MyCode(image=image, normalize=normalize, quotient=quotient).execute(constant=constant, threshold=threshold)
    result = result * quotient if normalize else result

    # write image to file
    output_path = args.o if args.o else join(dirname(image_path), 'output.tif')
    output_path = write_image(file_path=output_path, image=result, geo_transform=metadata['geo_transform'],
                              projection=metadata['projection'])

    print("Written to file: {}".format(output_path))
    return output_path


def main():
    parser = create_parser()
    run_code(parser.parse_args())


if __name__ == '__main__':
    main()
