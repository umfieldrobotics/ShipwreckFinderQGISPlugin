import sys
import os

QGIS_PYTHON_INSTALL = '/usr/share/qgis/python/'
WORKSPACE_PATH = os.getenv("HOME") + '/Research/noaa-multibeam/noaa-multibeam-qgis/'

import csv
import cv2
import math
import numpy as np

from dataclasses import dataclass

import pvl
import pdr

from osgeo import gdal

sys.path.append(QGIS_PYTHON_INSTALL)                  # PyQGIS path

from qgis.core import *
from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry
from qgis.utils import iface

class InferencePlugin:
    def __init__(self):
        self.PROJ_PATH = 'base.qgz' # base QGIS project with georeferenced aerial maps



if __name__=='__main__':

    gdal.UseExceptions()

    im = InferencePlugin()

    ###################
    ### Set up QGIS ###
    ###################

    print("----------SETTING UP QGIS----------")

    QgsApplication.setPrefixPath("/usr", True)

    qgs = QgsApplication([], True)
    qgs.initQgis()

    project = QgsProject.instance()
    project.read(WORKSPACE_PATH + im.PROJ_PATH) # if we want to load a project other than what is open

    ###########################################
    ### Grab current extent of active layer ###
    ###########################################
    
    layer = iface.activeLayer() # load the layer as you want
    ext = layer.extent()
    
    iface.mapCanvas().saveAsImage('./test.png')
    
    sys.stdout.flush()
    qgs.exitQgis()
