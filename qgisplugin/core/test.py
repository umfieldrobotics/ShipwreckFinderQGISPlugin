import torch 
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os 
    
from qgisplugin.core.models import *
from qgisplugin.core.smooth_tiled_predictions import predict_img_with_smooth_windowing


def crop_center(image, crop_height=512, crop_width=512):
    """
    Crops the center of the image to the specified dimensions.

    Parameters:
        image (np.ndarray): The input image array.
        crop_height (int): The height of the crop. Default is 512.
        crop_width (int): The width of the crop. Default is 512.

    Returns:
        np.ndarray: The cropped image array.
    """
    height, width = image.shape[:2]
    start_y = (height - crop_height) // 2
    start_x = (width - crop_width) // 2
    return image[start_y:start_y + crop_height, start_x:start_x + crop_width]

def normalize_image(img):
    #normalize the image
    img = (img - img.mean())/img.std()
    img = torch.from_numpy(img[..., 0]).float().unsqueeze(1) #.unsqueeze(0)

    return img

def test_with_chunk_blending(test_img_path, weight_path):
    '''
    Segmentation test by overlapping image chunks and merging
    using gaussian smoothing. From this git repo:
    https://github.com/Vooban/Smoothly-Blend-Image-Patches/tree/master
    '''
    model = Unet(1, 2)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model #.cuda()

    image = Image.open(test_img_path)
    og_image = np.array(image)
    gray_image = np.dot(og_image[..., :3], [0.2989, 0.5870, 0.1140])
    image = np.expand_dims(gray_image, axis=-1)

    # Make smooth predictions using overlapping tiles
    predictions_smooth = predict_img_with_smooth_windowing(
        image, 
        window_size=501,
        subdivisions=2,
        nb_classes=2,
        pred_func=(
            lambda img_batch_subdiv: model(normalize_image(img_batch_subdiv))
        )
    )

    predictions_smooth = torch.from_numpy(predictions_smooth)
    predictions_smooth = predictions_smooth.argmax(dim=2)
    predictions_smooth = predictions_smooth.cpu().detach().numpy()
    predictions_smooth = np.expand_dims(predictions_smooth, axis=0)
    predictions_smooth = np.squeeze(predictions_smooth)
    output_file_name = "smooth_out.png"
    plt.imsave(output_file_name, predictions_smooth, cmap="jet")




def test(test_files, weight_path):
    '''NAIVE chunking test function
    
    @params
    test_files: path to .tiff files that should be run through the model

    '''
    #load the model 
    model = Unet(1, 2)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model #.cuda()


    output_tiff_file_names = []
    output_numpy_file_names = []

    #read all the images using PIL 
    for test_file in test_files:
        image = Image.open(test_file)#np.load(test_file)
        output_file_name = test_file.replace(".tiff", ".tif").replace(".tif", "_pred.tiff")

        #resize to 501x501 
        image = image.resize((501, 501))

        og_image = np.array(image)
        gray_image = np.dot(og_image[..., :3], [0.2989, 0.5870, 0.1140]) #TODO: What are these magic numbers
        image = gray_image
        #normalize the image using the mean and std 
        image = (image - image.mean())/image.std()
        #threshold anything beyond 3 stds
        image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)#.cuda()

        #run the model on the image 
        pred = model(image)

        # Save the non-thresholded image to npy array
        pred_numpy = pred.detach().cpu().numpy()
        pred_numpy_filename = os.path.splitext(output_file_name)[0] + ".npy"
        np.save(pred_numpy_filename, pred_numpy)
        output_numpy_file_names.append(pred_numpy_filename)

        pred = pred.argmax(dim=1)
        pred = pred.cpu().detach().numpy()
        pred = np.expand_dims(pred, axis=0)
        pred = np.squeeze(pred)

        #create an array with the image and the prediction overlayed
        # pred_mask = np.zeros((501, 501, 3))
        # pred_mask[...,1] = pred
        # overlay = np.zeros((501, 501))
        # overlay = 0.8*og_image/255.0 + 0.2*pred_mask
        # plt.imshow(pred_mask)

        #save the image to the same directory 
        plt.imsave(output_file_name, pred, cmap="jet")
        output_tiff_file_names.append(output_file_name)
        

    return output_tiff_file_names, output_numpy_file_names

def main(): 
    # train("./", 1000, 1e-4)
    test("/home/frog/dev/model/images", "/home/frog/dev/model/mbes_unet.pt")


if __name__ == "__main__":
    main()