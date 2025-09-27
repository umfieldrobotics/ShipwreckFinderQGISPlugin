from ..safe_libs_setup import setup_libs, safe_import_ml_libraries

setup_libs()
libs = safe_import_ml_libraries()


import torch 

from models import *
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import wandb
import numpy as np
import glob 

from smooth_tiled_predictions import predict_img_with_smooth_windowing


SHOW_DEGUG_IMAGES = True

def normalize_image(img):
    #normalize the image
    img = (img - img.mean())/img.std()
    img = torch.from_numpy(img[..., 0]).float().unsqueeze(1) #.unsqueeze(0)

    return img


# Load the model
model = Unet(1, 2)
model.load_state_dict(torch.load("mbes_unet.pt", map_location=torch.device('cpu')))

# Load the input image
image = Image.open("new_filled_image.png")
# image = image.resize((900, 900))
og_image = np.array(image)
gray_image = np.dot(og_image[..., :3], [0.2989, 0.5870, 0.1140])
image = np.expand_dims(gray_image, axis=-1)
print(f"The image shape is {image.shape}")

if SHOW_DEGUG_IMAGES:
    # For demo purpose, let's look once at the window:
    plt.imshow(image)
    plt.title("This is the input image")
    plt.show()



predictions_smooth = predict_img_with_smooth_windowing(
    image,
    window_size=500,
    subdivisions=2,
    nb_classes=2,
    pred_func=(
        lambda img_batch_subdiv: model(normalize_image(img_batch_subdiv))
    )
)
print("we are finished whoop whoop!")
np.save("smoothed_numpy_array.npy", predictions_smooth)
print("The final shape is:", predictions_smooth.shape)
predictions_smooth = torch.from_numpy(predictions_smooth)
predictions_smooth = predictions_smooth.argmax(dim=2)
predictions_smooth = predictions_smooth.cpu().detach().numpy()
predictions_smooth = np.expand_dims(predictions_smooth, axis=0)
predictions_smooth = np.squeeze(predictions_smooth)

output_file_name = "out.png"
plt.imsave(output_file_name, predictions_smooth, cmap="jet")
