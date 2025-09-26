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
import sys
def setup_libs():
    current_dir = os.path.dirname(__file__)
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, 'libs')):
            libs_dir = os.path.join(current_dir, 'libs')
            if libs_dir not in sys.path:
                sys.path.insert(0, libs_dir)
            return
        current_dir = os.path.dirname(current_dir)
setup_libs()


from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer, QgsProcessingParameterNumber, \
    QgsProcessingParameterFileDestination, QgsProcessingParameterBoolean, QgsProcessingContext

from qgisplugin.core.ship_seeker import ShipSeeker
from qgisplugin.interfaces import import_image, write_image


class MyProcessingAlgorithm(QgsProcessingAlgorithm):
    # Constants used to refer to parameters and outputs.
    # They will be used when calling the algorithm from another algorithm, or when calling from the QGIS console.
    INPUT = 'INPUT IMAGE'
    NORMALIZE = 'NORMALIZE BOOLEAN'
    NORMALIZATION_VALUE = 'NORMALIZATION QUOTIENT'
    CONSTANT = 'CONSTANT'
    THRESHOLD = 'THRESHOLD'
    OUTPUT = 'OUTPUT FILE'

    @staticmethod
    def tr(string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return MyProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This string should be: fixed for the algorithm;
        not localised; unique within each provider; lowercase alphanumeric char only; no spaces or other formatting char
        """
        return 'my_plugin_processing'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any user-visible display of the algorithm name.
        """
        return self.tr('Shipwreck Seeker')

    def icon(self):
        """
        Should return a QIcon which is used for your provider inside the Processing toolbox.
        """
        return QIcon(':/plugin_logo')

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string should provide a basic description about
        what the algorithm does and the parameters and outputs associated with it.
        """
        return self.tr("Short description of your plugin and of the variables to show in the right panel.")

    def initAlgorithm(self, configuration, p_str=None, Any=None, *args, **kwargs):
        """
        Here we define the inputs and output of the algorithm, along with some other properties.
        """

        # We add the input image
        self.addParameter(QgsProcessingParameterRasterLayer(
            name=self.INPUT, description=self.tr('Image'))
        )

        self.addParameter(QgsProcessingParameterBoolean(
            name=self.NORMALIZE, description=self.tr('Normalize image?'), defaultValue=False)
        )

        self.addParameter(QgsProcessingParameterNumber(
            name=self.NORMALIZATION_VALUE, description=self.tr('Normalization quotient'), defaultValue=255,
            minValue=1, maxValue=1000000000, type=QgsProcessingParameterNumber.Double)  # type=1 is Double, 0 is int
        )

        param_constant = QgsProcessingParameterNumber(
            name=self.CONSTANT, description=self.tr('Constant to add to the image'), defaultValue=0.01,
            minValue=0.001, maxValue=1000000.000, type=QgsProcessingParameterNumber.Double  # type=1 is Double, 0 is int
        )

        param_constant.setMetadata({'widget_wrapper': {'decimals': 3}})
        self.addParameter(param_constant)

        param_threshold = QgsProcessingParameterNumber(
            name=self.THRESHOLD, description=self.tr('Threshold to set cells to 0'), defaultValue=0.2,
            minValue=0.001, maxValue=1000000.000, type=QgsProcessingParameterNumber.Double  # type=1 is Double, 0 is int
        )

        param_threshold.setMetadata({'widget_wrapper': {'decimals': 6}})
        self.addParameter(param_threshold)

        self.addParameter(QgsProcessingParameterFileDestination(
            name=self.OUTPUT, description=self.tr('Output file'))
        )

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        # todo: read all parameters, throw errors when needed, give user feedback and run code

        # Read input values
        image_path = self.parameterAsRasterLayer(parameters, self.INPUT, context).source()
        image, metadata = import_image(image_path)
        normalize = self.parameterAsBoolean(parameters, self.NORMALIZE, context),
        quotient = self.parameterAsDouble(parameters, self.NORMALIZATION_VALUE, context)

        result = MyCode(
            image=image,
            normalize=normalize,
            quotient=quotient
        ).execute(
            constant=self.parameterAsDouble(parameters, self.CONSTANT, context),
            threshold=self.parameterAsDouble(parameters, self.THRESHOLD, context),
            set_progress=feedback.setProgress,
            log=feedback.pushInfo
        )
        result = result * quotient if normalize else result
        feedback.setProgress(100)

        # Return outputs
        output_path = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        output_path = write_image(file_path=output_path, image=result, geo_transform=metadata['geo_transform'],
                                  projection=metadata['projection'])

        feedback.pushInfo("Written to file: {}".format(output_path))
        context.addLayerToLoadOnCompletion(output_path, QgsProcessingContext.LayerDetails(name='Processed Image',
                                                                                          project=context.project()))

        return {'Processed Image': output_path}
