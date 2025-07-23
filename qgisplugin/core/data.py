import cv2
import numpy as np
import os
import torch

from torch.utils.data import Dataset
from torchvision import transforms
import rasterio
from PIL import Image

def normalize_nonzero(image):
    """
    Normalizes nonzero values in an image (tensor or numpy array) to the range [0,1].
    
    Parameters:
        image (torch.Tensor or np.ndarray): Input image.

    Returns:
        torch.Tensor or np.ndarray: Normalized image with nonzero values scaled to [0,1].
    """
    is_tensor = isinstance(image, torch.Tensor)
    # print("is tensor", is_tensor)
    
    if not (is_tensor or isinstance(image, np.ndarray)):
        raise TypeError("Input must be a PyTorch tensor or a NumPy array.")
    
    nonzero_mask = image != 0  # Boolean mask for nonzero elements
    
    if is_tensor:
        if torch.any(nonzero_mask):  # Ensure there are nonzero values
            image_min = torch.min(image[nonzero_mask])
            image_max = torch.max(image[nonzero_mask])

            if not (image_min >= 0 and image_max <= 1):  # Only normalize if needed
                image[nonzero_mask] = (image[nonzero_mask] - image_min) / (image_max - image_min)
    
    else:  # NumPy array case
        if np.any(nonzero_mask):  # Ensure there are nonzero values
            image_min = np.min(image[nonzero_mask])
            image_max = np.max(image[nonzero_mask])

            if not (image_min >= 0 and image_max <= 1):  # Only normalize if needed
                image[nonzero_mask] = (image[nonzero_mask] - image_min) / (image_max - image_min)
    
    return image

class MBESDataset(Dataset):
    def __init__(self, root_dir, ignore_files, transform=None, byt=False, aug_multiplier=0, using_hillshade=False, using_inpainted=False, resize_to_div_16=False):
        self.root_dir = root_dir
        self.transform = transform
        self.byt = byt
        self.aug_multiplier = aug_multiplier  # Number of additional augmented samples per image
        self.img_size = 512 if resize_to_div_16 else 400 # SET TO 400 IN THE FUTURE
        
        self.using_hillshade = using_hillshade
        self.using_inpainted = using_inpainted

        self.resize = transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)  # Resize to 501 x 501

        # if self.using_inpainted:
        #     self.file_list = [file_name for file_name in os.listdir(os.path.join(root_dir, "inpainted")) if "_image.npy" in file_name]
        #     self.original_file_list = [file_name for file_name in os.listdir(os.path.join(root_dir, "original")) if "_image.npy" in file_name] # non inpainted files, we need these to get the mask for the invalid pixels
        # else:
        #     self.file_list = [file_name for file_name in os.listdir(root_dir) if "_image.npy" in file_name]

        # self.expanded_file_list = [(file_name, 0) for file_name in os.listdir(self.root_dir) if file_name not in ignore_files]
        self.expanded_file_list = [(os.path.join(self.root_dir, file_name), 0) for file_name in os.listdir(self.root_dir) \
                                    if ".npy" in file_name and os.path.join(self.root_dir, file_name) not in ignore_files]

        # with open("/home/smitd/Desktop/file_list.txt", 'w') as f:
        #     for elt in self.expanded_file_list:
        #         f.write(f"{elt[0]}\n")
        
        # with open("/home/smitd/Desktop/listdir.txt", 'w') as f:
        #     for file in os.listdir(self.root_dir):
        #         f.write(f"{file}\n")

        # import time
        # time.sleep(60)
        

        # self.expanded_file_list = [(file_name, i) for file_name in self.file_list for i in range(aug_multiplier + 1) if file_name not in ignore_files]
        
        # if self.using_inpainted:
        #     self.original_expanded_file_list = [(file_name, i) for file_name in self.original_file_list for i in range(aug_multiplier + 1)]

    def __len__(self):
        return len(self.expanded_file_list)
 
    def __getitem__(self, idx):
        file_name, _ = self.expanded_file_list[idx]
        image_name = file_name # os.path.join(self.root_dir, file_name)
        label_name = os.path.splitext(file_name)[0] + ".tif" # "Label" is the original chunk tif used for metadata

        # print(f"Image name: {image_name}")
        
        # if self.using_inpainted:
        #     original_file_name, _ = self.original_expanded_file_list[idx]
        #     original_image_name = os.path.join(self.root_dir, "original", original_file_name)

        # --- Load image ---
        # image = torch.from_numpy(np.load(image_name)).float()  # (H, W)
        og_image = np.load(image_name)
        # og_image = Image.open(image_name) # .convert('L')
        # og_image = np.array(og_image)
        # with rasterio.open(image_name) as src:
        #     og_image = src.read(1)

        # print(f"\t\t\tmin: {np.min(og_image)}, max: {np.max(og_image)}")

        if self.using_inpainted: # Compute inpainted image
            if og_image.ndim == 3:
                og_image = og_image[0] # Take first channel
            # extreme_nodata_mask = (og_image < -1000)
            # og_image[extreme_nodata_mask] = -2000
            inpaint_mask = ((og_image >= 1000) | (og_image <= -1000)).astype(np.uint8)
            og_image = cv2.inpaint(og_image.astype(np.float32), inpaint_mask, inpaintRadius=8, flags=cv2.INPAINT_NS)

        # print(f"\t\t\tmin: {np.min(og_image)}, max: {np.max(og_image)}")

        # np.save(os.path.join("/home/smitd/Documents/Inpainted_chunks", file_name), og_image)

        image = torch.from_numpy(og_image).float()

        if image.ndim == 2:
            image = image.unsqueeze(0)  # (1, H, W)

        image = self.resize(image)  # (1, H, W)
        mask = (image[0] == 0)
        image[0] = normalize_nonzero(image[0])

        # if self.using_inpainted: # load original (not inpainted) image and use that to generate the maskfor invalid regions
        #     original_image = torch.from_numpy(np.load(original_image_name)).float()  # (H, W)
        #     if original_image.ndim == 2:
        #         original_image = original_image.unsqueeze(0)  # (1, H, W)
            
        #     original_image = self.resize(original_image)  # (1, H, W)
        #     original_mask = (original_image[0] == 0)
        #     original_image[0] = normalize_nonzero(original_image[0])
            
        #     mask = original_mask

        # --- Load label ---
        # if os.path.exists(label_name):
        #     label_np = np.load(label_name).astype(np.int32) > 0
        # else:
        #     label_np = np.zeros((self.img_size, self.img_size), dtype=np.int32)
        label_np = np.zeros((self.img_size, self.img_size), dtype=np.int32)

        # Resize using nearest neighbor to preserve {-1, 0, 1}
        label = torch.from_numpy(label_np).unsqueeze(0).unsqueeze(0)  # (1, H, W)
        label = torch.nn.functional.interpolate(label.float(), size=(self.img_size, self.img_size), mode='nearest').squeeze(0).squeeze(0).long()  # (H, W)

        # Temporarily turn -1 → 255 for Albumentations
        label[label == -1] = 255

        # --- Optional hillshade ---
        if self.using_hillshade:
            hillshade_file_name, _ = self.expanded_file_list[idx]
            hillshade_path = os.path.join(self.root_dir, "hillshade", hillshade_file_name)
            hillshade = torch.from_numpy(np.load(hillshade_path)).float().unsqueeze(0)  # (1, H, W)
            hillshade = self.resize(hillshade) / 255.0
            image = torch.cat([image, hillshade], dim=0)  # (2, H, W)

        # --- Albumentations ---
        image_npy = image.permute(1, 2, 0).numpy().astype(np.float32)  # (H, W, C)
        mask_npy = (mask.numpy() * 255).astype(np.int32)
        label_npy = label.numpy().astype(np.int32)
        masks = [mask_npy, label_npy]

        if self.transform:
            transformed = self.transform(image=image_npy, masks=masks)
            image_npy = transformed["image"]
            masks = transformed["masks"]

        # Convert back to torch
        image = torch.tensor(image_npy).permute(2, 0, 1).float()
        transformed_mask = torch.tensor(masks[0], dtype=torch.long)
        label = torch.tensor(masks[1], dtype=torch.long)

        # Post-transform fix: 255 back to -1
        label[label == 255] = -1
        label[transformed_mask == 255] = -1

        # print(f"Image none: {image is None}")
        # print(f"Label none: {label is None}")

        return {
            'image': image,
            'label': label,
            'metadata': {
                "image_name": image_name,
                "label_name": label_name
            }
        }


        # Save images to confirm augmentations
        # np.save(os.path.join('QGIS_Chunks', os.path.basename(file_name)), image) # Save image
        # save_plot(os.path.join('Augmented_Ships', os.path.basename(file_name.replace("_image.npy", "_normalized.png"))), image)
        # if self.transform:
        #     img = Image.fromarray((255*image).astype(np.uint8))
        #     img.save(os.path.join('Augmented_Ships', os.path.basename(file_name.replace("_image.npy", "_augmented.png")))) # Save image
        #     lab = Image.fromarray((127.5*label.numpy()+127.5).astype(np.uint8)) # scaled for viz purposes 
        #     lab.save(os.path.join('Augmented_Ships', os.path.basename(file_name.replace("_image.npy", "_augmented_label.png")))) # Save image

        return {'image': image, 'label': label, 'metadata': {"image_name": image_name, "label_name": label_name}}
    
    def compute_hillshade(self, elevation, azimuth=315, altitude=45, cell_size=1.0):
        """
        Generate hillshade from a north-up aligned elevation array, preserving size.

        Parameters:
            elevation (ndarray): 2D NumPy array of elevation values.
            azimuth (float): Sun azimuth in degrees (clockwise from north).
            altitude (float): Sun altitude angle in degrees above horizon.
            cell_size (float): Spatial resolution in both x and y directions.

        Returns:
            hillshade (ndarray): 2D hillshade image (uint8), same shape as input.
        """
        # Convert angles to radians
        azimuth_rad = np.radians(360.0 - azimuth + 90.0)
        altitude_rad = np.radians(altitude)

        # Pad elevation to avoid edge loss
        padded = np.pad(elevation, pad_width=1, mode='edge')

        # Compute gradients (Horn's method)
        dzdx = ((padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * cell_size))
        dzdy = ((padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * cell_size))

        # Compute slope and aspect
        slope = np.arctan(np.hypot(dzdx, dzdy))
        aspect = np.arctan2(dzdy, -dzdx)
        aspect = np.where(aspect < 0, 2 * np.pi + aspect, aspect)

        # Illumination from the sun
        shaded = (
            np.sin(altitude_rad) * np.cos(slope) +
            np.cos(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
        )

        hillshade = np.clip(shaded, 0, 1) * 255
        return hillshade.astype(np.uint8)