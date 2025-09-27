# -*- coding: utf-8 -*-
"""
 This script initializes the plugin, making it known to QGIS.
"""

from .safe_libs_setup import setup_libs, safe_import_ml_libraries

setup_libs()
libs = safe_import_ml_libraries()


# noinspection PyPep8Naming
def classFactory(iface):
    """ Load NeuralNetworkPlugin class.
    :param QgsInterface iface: A QGIS interface instance.
    """
    from qgisplugin.shipwreck_seeker import ShipwreckSeeker
    # todo: update with correct plugin name if you changed it
    return ShipwreckSeeker(iface)
