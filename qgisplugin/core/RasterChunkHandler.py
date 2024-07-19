from osgeo import gdal, osr
import numpy as np

from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject, QgsPointXY, QgsRasterLayer, QgsRectangle

class RasterChunkHandler:
    '''This class abstracts getting specified chunks from a raster image...'''

    # can get using self.aster_layer.dataProvider().dataSourceUri()
    def __init__(self, raster_path: str, chunk_size = 501):

        self.raster_path = raster_path
        self.chunk_size = chunk_size

        url = self.raster_path.split('|')[0]
        options = self.raster_path.split('|')[1:]
        options[0].replace("option:", "")

        self.dataset = gdal.OpenEx(url, open_options=options)

        self.num_bands = self.dataset.RasterCount
        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize

    def get_chunk_as_np_arr(self, chunk_x, chunk_y):

        x_offset, y_offset = self.get_chunk_offset(chunk_x, chunk_y)

        all_bands = []
        for band_num in range(1, self.num_bands + 1):
            band = self.dataset.GetRasterBand(band_num)

            data = band.ReadAsArray(x_offset, y_offset, self.chunk_size, self.chunk_size)

            # PAD the data
            new_data = np.zeros((501, 501))
            new_data[:len(data), :len(data[0])] = data
            all_bands.append(new_data)
        return np.stack(all_bands)
    
    def get_num_x_chunks(self):
        return np.ceil(self.width * 1.0 / self.chunk_size)

    def get_num_y_chunks(self):
        return np.ceil(self.height * 1.0 / self.chunk_size)

    def get_chunk_offset(self, chunk_x, chunk_y):
        """
        Calculate the pixel offset for a specific chunk.
        
        :param chunk_x: X index of the chunk.
        :param chunk_y: Y index of the chunk.
        :return: Tuple (x_offset, y_offset) in pixels.
        """
        width, height = self.chunk_size, self.chunk_size
        x_offset = chunk_x * width
        y_offset = chunk_y * height

        return x_offset, y_offset

            
