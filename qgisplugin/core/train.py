import torch 
import cv2

from qgisplugin.core.models import *
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torch.autograd import Variable
import wandb
import numpy as np
import os 
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


# BASNET TEST FUNCTION
def basnet_test(test_path, ignore_files, weight_path, chunk_size, cell_size, set_progress: callable=None):
    threshold = 0.1

    # Load the model
    model = BASNet(3, 1)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.cuda()

    model.eval()

    test_dataset = MBESDataset(test_path, ignore_files, using_hillshade=False, using_inpainted=True, resize_to_div_16=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # print("Post dataloader")

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            image1 = data['image'].type(torch.FloatTensor)
            image = torch.hstack([image1, image1, image1])
            image_file_path = data['metadata']['label_name'][0] # "Label" is the corresponding .tif file for metadata
            output_file_name = image_file_path.replace(".tiff", ".tif").replace(".tif", "_pred.tiff")

            # save_name = os.path.splitext(os.path.basename(image_file_path))[0] + ".png"
            # cv2.imwrite(os.path.join("/home/smitd/Documents/Copied_Temp_Chunks/saved_getitem", save_name), image1.cpu().numpy().squeeze()*255)
            
            image_v = Variable(image, requires_grad=True).cuda()

            _, d1, _, _, _, _, _, _ = model(image_v)

            pred = d1[:,0,:,:]
            pred = normPRED(pred)
            
            # Undo the resize
            resize = transforms.Resize((chunk_size, chunk_size), interpolation=transforms.InterpolationMode.NEAREST)
            pred = resize(pred)

            pred = pred.cpu().detach().numpy()
            
            # pred_numpy_filename = os.path.splitext(output_file_name)[0] + ".npy"
            # np.save(pred_numpy_filename, pred)

            pred = (pred >= threshold).astype(np.int32)
            pred = np.expand_dims(pred, axis=1)
            pred = np.squeeze(pred)

            # cv2.imwrite(os.path.join("/home/smitd/Documents/Copied_Temp_Chunks/saved_model_out", "postthresh" + save_name), pred*255)

            # # Remove small contours
            # min_area = int((chunk_size**2) * 0.0006)  # adjust this value as needed
            # pred = pred.astype(np.uint8)
            # contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # cleaned = np.zeros_like(pred)
            # for cnt in contours:
            #     area = cv2.contourArea(cnt)
            #     print(f"Contour area: {area}")
            #     if area >= min_area:
            #         cv2.drawContours(cleaned, [cnt], -1, 1, thickness=cv2.FILLED)
            #     # else:
            #     #     print(f"Removing contour with area {area}")
            # pred = cleaned.astype(np.int32)

            
            # cv2.imwrite(os.path.join("/home/smitd/Documents/Copied_Temp_Chunks/saved_model_out", "postclean" + save_name), pred*255)
            
            plt.imsave(output_file_name, pred, cmap="jet")
            copy_tiff_metadata(image_file_path, output_file_name)

            set_progress(20 + int((i * 60) // len(test_loader.dataset)))



# UNET TEST FUNCTION
def unet_test(test_file_dir, ignore_files, weight_path, chunk_size, cell_size, set_progress: callable = None, hillshade = True):
    #load the model

    if hillshade:
        # FOR TWO CHANNEL INPUT
        model = Unet(2, 2)
    else:
        # FOR ONE CHANNEL INPUT
        model = Unet(1, 2)
    
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.cuda()
    # model.cpu()

    model.eval()

    # print(f"Model device: {model.resnet_encoder.conv1.device}")

    # output_tiff_file_names = []
    # output_numpy_file_names = []

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
            image = data['image'].cuda()
            image_file_path = data['metadata']['label_name'][0] # "Label" is the corresponding .tif file for metadata

            save_name = os.path.splitext(os.path.basename(image_file_path))[0] + ".png"
            # # print(image.cpu().numpy())
            # print("Saving chunk")
            # cv2.imwrite(os.path.join("/home/smitd/Documents/Copied_Temp_Chunks/saved_getitem", "layer0_" + save_name), image.cpu().numpy().squeeze()[0]*255)
            # cv2.imwrite(os.path.join("/home/smitd/Documents/Copied_Temp_Chunks/saved_getitem", "layer1_" + save_name), image.cpu().numpy().squeeze()[1]*255)

            output_file_name = image_file_path.replace(".tiff", ".tif").replace(".tif", "_pred.tiff")

            # print(f"Min: {image.min()}, Max: {image.max()}, Size: {image.shape}")
            # print(f"Device: {next(model.parameters()).device}")
            # Run through model
            pred = model(image)

            # pred2 = pred.argmax(dim=1)
            # pred2= pred2.cpu().detach().numpy()
            # pred2 = np.expand_dims(pred2, axis=0)
            # pred2 = np.squeeze(pred2)
            # cv2.imwrite(os.path.join("/home/smitd/Documents/Copied_Temp_Chunks/saved_model_out", "freshout" + save_name), pred2*255)

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

            # print(f"Prediction shape: {pred.shape}")
            # cv2.imwrite(os.path.join("/home/smitd/Documents/Copied_Temp_Chunks/saved_model_out", "post" + save_name), pred*255)

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

    # print("conv1 shape:", model.conv1.weight.shape)

    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.cuda()
    model.eval()

    dataset = MBESDataset(test_file_dir, ignore_files, using_hillshade=False, using_inpainted=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            image = data['image'].cuda()
            # image = torch.hstack([image, image, image])
            image_file_path = data['metadata']['label_name'][0]
            output_file_name = image_file_path.replace(".tiff", ".tif").replace(".tif", "_pred.tiff")


            # save_name = os.path.splitext(os.path.basename(image_file_path))[0] + ".png"
            # print(image.cpu().numpy())
            # print("Saving chunk")
            # cv2.imwrite(os.path.join("/home/smitd/Documents/Copied_Temp_Chunks/saved_getitem", save_name), image.cpu().numpy().squeeze()*255)

            pred = model(image)[0]
            # print(f"Pred shape before interpolate: {pred.shape}")
            pred = F.interpolate(pred, size=image.shape[2:], mode='bilinear', align_corners=True)
            # print(f"Post model shape: {pred.shape}")
            pred = pred.argmax(dim=1)

            # Undo the resize
            resize = transforms.Resize((chunk_size, chunk_size), interpolation=transforms.InterpolationMode.NEAREST)
            pred = resize(pred)

            pred = pred.cpu().detach().numpy()
            pred = np.squeeze(pred)

            # print(f"Prediction shape pre-contour: {pred.shape}")

            # cv2.imwrite(os.path.join("/home/smitd/Documents/Copied_Temp_Chunks/saved_model_out", "preclean_" + save_name), pred*255)
            
            # Remove small contours
            # min_area = 150  # adjust this value as needed

            # pred = pred.astype(np.uint8)
            # contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cleaned = np.zeros_like(pred)
            # for cnt in contours:
            #     area = cv2.contourArea(cnt)
            #     if area >= min_area:
            #         cv2.drawContours(cleaned, [cnt], -1, 1, thickness=cv2.FILLED)
            #     # else:
            #     #     print(f"Removing contour with area {area}")
            # pred = cleaned.astype(np.int32)

            # cv2.imwrite(os.path.join("/home/smitd/Documents/Copied_Temp_Chunks/saved_model_out", "postclean_" + save_name), pred*255)

            # print(f"Prediction shape post-contour: {pred.shape}")

            # Save tiff and copy metadata
            plt.imsave(output_file_name, pred, cmap="jet")
            copy_tiff_metadata(image_file_path, output_file_name)

            set_progress(20 + int((i * 60) // len(dataloader.dataset)))

def main(): 
    # train("./", 1000, 1e-4)
    test("/home/frog/dev/model/images", "/home/frog/dev/model/mbes_unet.pt")


if __name__ == "__main__":
    main()