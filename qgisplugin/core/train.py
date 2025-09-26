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


import torch 
import cv2

from qgisplugin.core.models import *
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.autograd import Variable
import wandb
import numpy as np
import glob 
import argparse
from qgisplugin.core.data import MBESDataset
from qgisplugin.core.tiff_utils import copy_tiff_metadata


from qgisplugin.core.hrnet.seg_hrnet_ocr import *
from qgisplugin.core.hrnet.config import config, update_config

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, byt=False):
        self.root_dir = root_dir
        self.transform = transform
        self.byt = byt
        self.file_list = [file_name for file_name in os.listdir(root_dir) if "_image.npy" in file_name]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.file_list[idx])
        label_name = image_name.replace("_image.npy", "_label.png.npy")

        image = torch.from_numpy(np.load(image_name)).float().unsqueeze(0)
        label = (torch.from_numpy(np.load(label_name)) > 0).long()

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
            # label = self.transform(label)

        return {'image': image, 'label': label}

def train(save_path, num_epochs, lr): 
    #open wandb 
    wandb.init(project="mbes")

    model = Unet(1, 2)
    model.cuda()

    dataset = CustomDataset("/mnt/syn/advaiths/mbes_data/data", byt=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    #find normalization constants for the timages 
    #mean = 0
    mean = 0.0
    std = 0.0

    for i, data in enumerate(dataloader):
        image = data['image'].cuda()
        label = data['label'].cuda()

        mean += image.mean()
        std += image.std()
    
    mean /= len(dataset)
    std /= len(dataset)

    #make tf trasnforms 
    tf = transforms.Compose([
        transforms.Normalize(mean.item(), std.item())
    ])

    dataset = CustomDataset("/mnt/syn/advaiths/mbes_data/data", transform=tf)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = torch.nn.CrossEntropyLoss()

    #write a train loop 
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        ep_loss  = []
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image = data['image'].cuda()
            label = data['label'].cuda()

            optim.zero_grad()
            pred = model(image)
            loss = ce_loss(pred, label)

            loss.backward()
            optim.step()
            ep_loss.append(loss.item())
        
        #log loss to wandb 
        wandb.log({"loss": np.mean(ep_loss)})
        if epoch % 10 == 0: 
            #log one of the images predicted 
            wandb.log({"image": wandb.Image(image[0].cpu().detach().numpy())})
            wandb.log({"label": wandb.Image(label[0].cpu().detach().numpy())})

            #colormap the segmentation prediction from the model
            pred = pred[0].cpu().detach().numpy()
            pred = np.argmax(pred, axis=0)
            pred = np.expand_dims(pred, axis=0)
            wandb.log({"pred": wandb.Image(pred)})


        torch.save(model.state_dict(), os.path.join(save_path, "model.pt".format(epoch)))

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

# BASNET TEST FUNCTION
def basnet_test(test_path, ignore_files, weight_path, chunk_size, cell_size, thresh=0.1, set_progress: callable=None):
    threshold = thresh

    # Load and prep the model
    model = BASNet(3, 1)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    model.eval()

    test_dataset = MBESDataset(test_path, ignore_files, using_hillshade=False, using_inpainted=True, resize_to_div_16=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            image1 = data['image'].type(torch.FloatTensor)
            image = torch.hstack([image1, image1, image1])
            image_file_path = data['metadata']['label_name'][0] # "Label" is the corresponding .tif file for metadata
            output_file_name = image_file_path.replace(".tiff", ".tif").replace(".tif", "_pred.tiff")

            
            image_v = Variable(image, requires_grad=True)

            if torch.cuda.is_available():
                image_v = image_v.cuda()
            else:
                image_v = image_v.cpu()

            _, d1, _, _, _, _, _, _ = model(image_v)

            pred = d1[:,0,:,:]
            pred = normPRED(pred)
            
            # Undo the resize
            resize = transforms.Resize((chunk_size, chunk_size), interpolation=transforms.InterpolationMode.NEAREST)
            pred = resize(pred)

            # Save the non-thresholded image to npy array
            pred_numpy = pred.detach().cpu().numpy()
            pred_numpy_filename = os.path.splitext(output_file_name)[0] + ".npy"
            np.save(pred_numpy_filename, pred_numpy)

            pred = pred.cpu().detach().numpy()

            pred = (pred >= threshold).astype(np.int32)
            pred = np.expand_dims(pred, axis=1)
            pred = np.squeeze(pred)
            
            plt.imsave(output_file_name, pred, cmap="jet")
            copy_tiff_metadata(image_file_path, output_file_name)

            set_progress(20 + int((i * 60) // len(test_loader.dataset)))

# UNET TEST FUNCTION
def unet_test(test_file_dir, ignore_files, weight_path, chunk_size, cell_size, set_progress: callable = None, hillshade = True):
    if hillshade:
        # FOR TWO CHANNEL INPUT
        model = Unet(2, 2)
    else:
        # FOR ONE CHANNEL INPUT
        model = Unet(1, 2)
    
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    model.eval()

    if hillshade:
        # FOR TWO CHANNEL
        dataset = MBESDataset(test_file_dir, ignore_files, using_hillshade=True, using_inpainted=True, cell_size=cell_size)
    else:
        # FOR ONE CHANNEL
        dataset = MBESDataset(test_file_dir, ignore_files, using_hillshade=False, using_inpainted=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # Get data and prep file paths
            image = data['image']

            if torch.cuda.is_available():
                image = image.cuda()
            else:
                image = image.cpu()
            
            image_file_path = data['metadata']['label_name'][0] # "Label" is the corresponding .tif file for metadata

            output_file_name = image_file_path.replace(".tiff", ".tif").replace(".tif", "_pred.tiff")

            # Run through model
            pred = model(image)

            # Undo the resize
            resize = transforms.Resize((chunk_size, chunk_size), interpolation=transforms.InterpolationMode.NEAREST)
            pred = resize(pred)

            # Save the non-thresholded image to npy array
            pred_numpy = pred.detach().cpu().numpy()
            pred_numpy_filename = os.path.splitext(output_file_name)[0] + ".npy"
            np.save(pred_numpy_filename, pred_numpy)

            pred = pred.argmax(dim=1)
            pred = pred.cpu().detach().numpy()
            pred = np.expand_dims(pred, axis=0)
            pred = np.squeeze(pred)

            # Save tiff and copy metadata
            plt.imsave(output_file_name, pred, cmap="jet")
            copy_tiff_metadata(image_file_path, output_file_name)

            set_progress(20 + int((i * 60) // len(dataloader.dataset)))

# HRNET TEST FUNCTION
def hrnet_test(test_file_dir, ignore_files, weight_path, chunk_size, cell_size, set_progress: callable = None):
    a = argparse.Namespace(cfg='hrnet/config/hrnet_config.py',
                                   local_rank=-1,
                                   opts=[],
                                   seed=304)
    update_config(config, a)

    model = get_seg_model(config)

    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    model.eval()

    dataset = MBESDataset(test_file_dir, ignore_files, using_hillshade=False, using_inpainted=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            image = data['image']

            if torch.cuda.is_available():
                image = image.cuda()
            else:
                image = image.cpu()

            image_file_path = data['metadata']['label_name'][0]
            output_file_name = image_file_path.replace(".tiff", ".tif").replace(".tif", "_pred.tiff")

            pred = model(image)[0]
            # print(f"Pred shape before interpolate: {pred.shape}")
            pred = F.interpolate(pred, size=image.shape[2:], mode='bilinear', align_corners=True)

            # Save the non-thresholded image to npy array
            pred_numpy = pred.detach().cpu().numpy()
            pred_numpy_filename = os.path.splitext(output_file_name)[0] + ".npy"
            np.save(pred_numpy_filename, pred_numpy)

            pred = pred.argmax(dim=1)

            # Undo the resize
            resize = transforms.Resize((chunk_size, chunk_size), interpolation=transforms.InterpolationMode.NEAREST)
            pred = resize(pred)

            pred = pred.cpu().detach().numpy()
            pred = np.squeeze(pred)

            # Save tiff and copy metadata
            plt.imsave(output_file_name, pred, cmap="jet")
            copy_tiff_metadata(image_file_path, output_file_name)

            set_progress(20 + int((i * 60) // len(dataloader.dataset)))

def main(): 
    # train("./", 1000, 1e-4)
    test("/home/frog/dev/model/images", "/home/frog/dev/model/mbes_unet.pt")


if __name__ == "__main__":
    main()