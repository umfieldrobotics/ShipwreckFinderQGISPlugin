# -*- coding: utf-8 -*-
"""
 This script initializes the plugin, making it known to QGIS.
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


# noinspection PyPep8Naming
def classFactory(iface):
    """ Load NeuralNetworkPlugin class.
    :param QgsInterface iface: A QGIS interface instance.
    """
    from qgisplugin.shipwreck_seeker import ShipwreckSeeker
    # todo: update with correct plugin name if you changed it
    return ShipwreckSeeker(iface)
