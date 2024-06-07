import torch 

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

def test(test_files, weight_path):
    #load the model 
    model = Unet(1, 2)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model #.cuda()


    output_file_names = []

    #read all the images using PIL 
    for test_file in test_files:
        image = Image.open(test_file)#np.load(test_file)

        #convert colormap from jet to grayscale using plt 
        # image = np.array(image)
        # image = crop_center(image, 501, 501)

        #resize to 501x501 
        image = image.resize((501, 501))

        og_image = np.array(image)
        gray_image = np.dot(og_image[..., :3], [0.2989, 0.5870, 0.1140])
        image = gray_image
        #normalize the image using the mean and std 
        image = (image - image.mean())/image.std()
        #threshold anything beyond 3 stds
        sanity = image > 3
        image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)#.cuda()

        #run the model on the image 
        pred = model(image)
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

        output_file_name = test_file.replace(".tiff", ".tif").replace(".tif", "_pred.tiff")
        plt.imsave(output_file_name, pred, cmap="jet")
        output_file_names.append(output_file_name)

    return output_file_names

def main(): 
    # train("./", 1000, 1e-4)
    test("/home/frog/dev/model/images", "/home/frog/dev/model/mbes_unet.pt")


if __name__ == "__main__":
    main()