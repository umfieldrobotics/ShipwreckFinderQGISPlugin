import numpy as np
import os

image = np.zeros((100,100))
print("Image shape", image.shape)
image_path = "/home/tylergs/Documents/noaa_multibeam_real_data/Training/Plugin_outputs"
np.save(os.path.join(image_path, str(3)), image)