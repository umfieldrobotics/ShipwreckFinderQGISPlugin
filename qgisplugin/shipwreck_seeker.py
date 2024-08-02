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
from os import path

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMenu
from qgis.core import QgsApplication

# todo: import your own GUI or PROVIDER:
from qgisplugin.interfaces.my_gui import MyWidget
from qgisplugin.interfaces.thresholding_gui import ThresholdingWidget
from qgisplugin.interfaces.preprocessing_gui import PreprocessingWidget
from qgisplugin.interfaces.my_plugin_provider import MyProcessingProvider
from qgisplugin.interfaces.extract_shipwrecks_gui import ExtractBoxesWidget
from qgisplugin.images.cqp_resources_rc import qInitResources
qInitResources()  # necessary to be able to access your images


import torch



class ShipwreckSeeker:
    """ QGIS Plugin Implementation """

    def __init__(self, iface):
        """
        :param QgsInterface iface: the interface instance which provides the hook to manipulate the QGIS GUI at run time
        """
        # Save reference to the QGIS interface
        self.iface = iface

        # initialize plugin directory
        self.plugin_dir = path.dirname(__file__)

        # Add an empty menu to the Raster Menu
        self.main_menu = QMenu(title='Shipwreck Seeker Menu', parent=self.iface.rasterMenu())
        self.main_menu.setIcon(QIcon(':/plugin_logo'))
        self.iface.rasterMenu().addMenu(self.main_menu)
        self.provider = None

    def initGui(self):
        """ Create the menu entries and toolbar icons inside the QGIS GUI """

        # add action button to raster menu
        action = QAction(QIcon(':/plugin_logo'), 'Shipwreck Seeker', self.iface.mainWindow())
        action.triggered.connect(self.run_shipseeker_widget)
        action.setStatusTip('Quick information on your plugin.')
        self.main_menu.addAction(action)

        # add another action button to raster menu
        action = QAction(QIcon(':/plugin_logo'), 'Threshold Model Output', self.iface.mainWindow())
        action.triggered.connect(self.run_thresholding_widget)
        action.setStatusTip('Quick information on your plugin.')
        self.main_menu.addAction(action)

        # add another action button to raster menu
        action = QAction(QIcon(':/plugin_logo'), 'Preprocess Raster Layer', self.iface.mainWindow())
        action.triggered.connect(self.run_preprocessing_widget)
        action.setStatusTip('Quick information on your plugin.')
        self.main_menu.addAction(action)

        action = QAction(QIcon(':/plugin_logo'), 'Extract Bounding Boxes', self.iface.mainWindow())
        action.triggered.connect(self.run_box_extractor_widget)
        action.setStatusTip('Quick information on your plugin.')
        self.main_menu.addAction(action)

        # add provider to processing toolbox
        self.provider = MyProcessingProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        self.iface.rasterMenu().removeAction(self.main_menu.menuAction())
        QgsApplication.processingRegistry().removeProvider(self.provider)

    @staticmethod
    def run_shipseeker_widget():
        widget = MyWidget()
        widget.show()
        widget.exec_()

    @staticmethod
    def run_thresholding_widget():
        widget = ThresholdingWidget()
        widget.show()
        widget.exec_()

    @staticmethod
    def run_preprocessing_widget():
        widget = PreprocessingWidget()
        widget.show()
        widget.exec_()


    @staticmethod
    def run_box_extractor_widget():
        widget = ExtractBoxesWidget()
        widget.show()
        widget.exec_()
    
