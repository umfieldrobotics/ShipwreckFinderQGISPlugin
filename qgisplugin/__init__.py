# -*- coding: utf-8 -*-
"""
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):
    """ Load NeuralNetworkPlugin class.
    :param QgsInterface iface: A QGIS interface instance.
    """
    from qgisplugin.shipwreck_seeker import ShipwreckSeeker
    # todo: update with correct plugin name if you changed it
    return ShipwreckSeeker(iface)
