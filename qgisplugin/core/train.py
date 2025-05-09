import torch 
import torch.nn.functional as F
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
from qgisplugin.core.utils import clear_directory 

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

    model = Unet(3, 2)
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

def test(test_files, weight_path, x_chunk_size, y_chunk_size):
    #load the model 
    model = Unet(3, 2)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval() #.cuda()
    print("chunk sizes:", x_chunk_size, y_chunk_size)

    output_tiff_file_names = []
    output_numpy_file_names = []
    
    image_path = "/home/tylergs/Documents/noaa_multibeam_real_data/Training/Plugin_outputs"
    #read all the images using PIL 
    for t, test_file in enumerate(test_files):
        image = Image.open(test_file)#np.load(test_file)
        output_file_name = test_file.replace(".tiff", ".tif").replace(".tif", "_pred.tiff")

        #convert colormap from jet to grayscale using plt 
        # image = np.array(image)
        # image = crop_center(image, 501, 501)

        # image = image.resize((x_chunk_size, y_chunk_size))
        image = image.resize((200,200))

        og_image = np.array(image)

        # New code for 3ch images
        # image_save_path = os.path.join(image_path, "OG_" + os.path.basename(test_file).replace(".tif", ""))
        # np.save(image_save_path, image) # 200x200x3, between 0 1nd 170

        # Normalize images between 0 and 1, not including 0 value pixels
        img = og_image.astype(np.float32)  # Convert to float32
        nonzero_mask = img != 0
        pct_valid = np.count_nonzero(nonzero_mask) / nonzero_mask.size
        if np.any(nonzero_mask):  # Ensure there are nonzero values
            img_min = img[nonzero_mask].min()
            img_max = img[nonzero_mask].max()
            print("Min, max:", img_min, img_max)
            
            if not (img_min >= 0 and img_max <= 1):  # Avoid division by zero
                img[nonzero_mask] = (img[nonzero_mask] - img_min) / (img_max - img_min)
                print("normalizing image:", test_file)

        image = torch.from_numpy(img).float().permute(2,0,1)

        color_save_path = os.path.join(image_path, "Color_" + os.path.basename(test_file).replace(".tif", "_image"))
        np.save(color_save_path, image.cpu().numpy()) # Masked, 200x200x3, between 0 and 1
        image = image.unsqueeze(0) # add batch dim 

        # Original code for 1ch images
        # gray_image = np.dot(og_image[..., :3], [0.2989, 0.5870, 0.1140]) #TODO: What are these magic numbers
        # image = gray_image
        # #normalize the image using the mean and std 
        # image = (image - image.mean())/image.std()
        # #threshold anything beyond 3 stds
        # sanity = image > 3

        # Write code to file so we can work with these outputs
        # gray_save_path = os.path.join(image_path, "Gray_" + os.path.basename(test_file))
        # # print("Image save path", image_save_path)
        # np.save(gray_save_path, image)
        # image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)#.cuda()


        #run the model on the image 
        pred = model(image)

        # print(image.mean(), pred.mean())

        # Save the non-thresholded image to npy array
        pred_numpy = pred.detach().cpu().numpy()
        pred_numpy_filename = os.path.splitext(output_file_name)[0] + ".npy"
        np.save(pred_numpy_filename, pred_numpy)
        output_numpy_file_names.append(pred_numpy_filename)

        pred = pred.argmax(dim=1)
        pred = pred.cpu().detach().numpy()
        # print("orig shape", pred.shape)
        # pred = np.expand_dims(pred, axis=0)
        pred = np.squeeze(pred)
        # apply mask
        zero_mask = img[..., 0] == 0
        pred[zero_mask] = 0
        pred = post_process(pred, pct_valid)

        print(f"Resize dimensions: ({x_chunk_size}, {y_chunk_size})")
        print("original shape: ", pred.copy().shape)
        if x_chunk_size > 0 and y_chunk_size > 0:
            pred = cv2.resize(pred.copy(), (x_chunk_size, y_chunk_size), interpolation=cv2.INTER_NEAREST)
        else:
            print(f"Invalid resize dimensions: ({x_chunk_size}, {y_chunk_size})")

        # print("after shape", pred.shape)

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

def post_process(image, valid): # image as np array
    if (valid) < 0.75:
        return np.zeros_like(image)
    return image

def test_new(test_path, weight_path, batch_size=1):
    # Load the model
    model = Unet(3, 2)
    model.load_state_dict(torch.load(weight_path))
    model.cuda()
    model.eval()
    # Load all the .npy files within test_path using glob
    test_files = glob.glob(os.path.join(test_path, "*_image.npy"))

    val_dataset = CustomDataset(test_path, byt=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    batch_count = 0
    for data in val_loader:
        image = data['image'].cuda()
        label = data['label'].cuda()

        pred = model(image)
        pred = pred.argmax(dim=1)

        label = label.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()

        label_flat = label.flatten()
        pred_flat = pred.flatten()


def main(): 
    # train("./", 1000, 1e-4)
    test("/home/frog/dev/model/images", "/home/frog/dev/model/mbes_unet.pt")


if __name__ == "__main__":
    main()