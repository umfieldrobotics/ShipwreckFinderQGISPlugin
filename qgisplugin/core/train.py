import torch 
import cv2

from qgisplugin.core.models import *
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import wandb
import numpy as np
import os 
import glob 
from qgisplugin.core.data import MBESDataset
from qgisplugin.core.tiff_utils import copy_tiff_metadata

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

        #shift the image by the min value 
        # if self.byt:
        #     image -= image.min()
        #     image = ((image/500.0)*255.0).to(torch.uint8) #scale 

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

            #calculate train miou for the entire tensor 
            # pred = pred.argmax(dim=1)
            # miou = torch.mean(iou(pred, label))
            # wandb.log({"miou": miou.item()})
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
        # print("Epoch: {}, Loss: {}".format(epoch, np.mean(ep_loss)))


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


def test(test_file_dir, ignore_files, weight_path, set_progress: callable = None):
    #load the model 
    # FOR ONE CHANNEL INPUT
    model = Unet(1, 2)
    # FOR TWO CHANNEL INPUT
    # model = Unet(2, 2)

    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # print(f"Model device: {model.resnet_encoder.conv1.device}")

    # output_tiff_file_names = []
    # output_numpy_file_names = []

    dataset = MBESDataset(test_file_dir, ignore_files, using_hillshade=False, using_inpainted=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # Get data and prep file paths
            image = data['image'].cuda()
            image_file_path = data['metadata']['label_name'][0] # "Label" is the corresponding .tif file for metadata

            # save_name = os.path.splitext(os.path.basename(image_file_path))[0] + ".png"
            # print(image.cpu().numpy())
            # print("Saving chunk")
            # cv2.imwrite(os.path.join("/home/smitd/Documents/Copied_Temp_Chunks/saved_getitem", save_name), image.cpu().numpy().squeeze()*255)

            output_file_name = image_file_path.replace(".tiff", ".tif").replace(".tif", "_pred.tiff")

            # print(f"Min: {image.min()}, Max: {image.max()}, Size: {image.shape}")

            # Run through model
            pred = model(image)

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

    # import time
    # time.sleep(20)

# # Load the model
#     model = Unet(1, 2)
#     model.load_state_dict(torch.load(weight_path))
#     model.cuda()

#     model.eval()

#     test_dataset = MBESDataset(test_path, byt=False, using_hillshade=False, using_inpainted=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

#     with torch.no_grad():
#         for data in test_loader:
#             image = data['image'].cuda()
#             label = data['label'].cuda()

#             pred = model(image)
#             pred = pred.argmax(dim=1)

#             label = label.cpu().detach().numpy()
#             pred = pred.cpu().detach().numpy()

#             valid_data_mask = (label.flatten() != -1)  # Ignore invalid pixels
#             label_flat = label.flatten()[valid_data_mask]
#             pred_flat = pred.flatten()[valid_data_mask]


def main(): 
    # train("./", 1000, 1e-4)
    test("/home/frog/dev/model/images", "/home/frog/dev/model/mbes_unet.pt")


if __name__ == "__main__":
    main()