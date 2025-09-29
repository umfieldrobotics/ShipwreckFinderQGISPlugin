import cv2
import numpy as np
import os
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from shipwreck_finder_demo.utils.utils import normalize_nonzero


class MBESDataset(Dataset):
    def __init__(self, root_dir, img_size=400, transform=None, byt=False, aug_multiplier=0, using_hillshade=False, using_inpainted=False, resize_to_div_16=False):
        self.root_dir = root_dir
        self.transform = transform
        self.byt = byt
        self.aug_multiplier = aug_multiplier  # Number of additional augmented samples per image
        self.img_size = img_size
        
        self.using_hillshade = using_hillshade
        self.using_inpainted = using_inpainted

        if self.using_inpainted:
            self.file_list = [file_name for file_name in os.listdir(os.path.join(root_dir, "inpainted")) if "_image.npy" in file_name]
            self.original_file_list = [file_name for file_name in os.listdir(os.path.join(root_dir, "original")) if "_image.npy" in file_name] # non inpainted files, we need these to get the mask for the invalid pixels
        else:
            self.file_list = [file_name for file_name in os.listdir(root_dir) if "_image.npy" in file_name]
        self.resize_dim = ((self.img_size // 32) + 1) * 32  if resize_to_div_16 else self.img_size
        self.resize = transforms.Resize((self.resize_dim, self.resize_dim), interpolation=transforms.InterpolationMode.NEAREST)

        self.expanded_file_list = [(file_name, i) for file_name in self.file_list for i in range(aug_multiplier + 1)]
        
        if self.using_inpainted:
            self.original_expanded_file_list = [(file_name, i) for file_name in self.original_file_list for i in range(aug_multiplier + 1)]

    def __len__(self):
        return len(self.expanded_file_list)
 
    def __getitem__(self, idx):
        file_name, _ = self.expanded_file_list[idx]
        image_name = os.path.join(self.root_dir, "inpainted", file_name) if self.using_inpainted else os.path.join(self.root_dir, file_name)
        label_name = image_name.replace("_image.npy", "_label.npy").replace("inpainted", "original")
        
        if self.using_inpainted:
            original_image_name = os.path.join(self.root_dir, "original", file_name)

        image = torch.from_numpy(np.load(image_name)).float()
        if image.ndim == 2:
            image = image.unsqueeze(0)

        image = self.resize(image)
        mask = (image[0] == 0)
        image[0] = normalize_nonzero(image[0])
        
        if self.using_inpainted: # load original (not inpainted) image and use that to generate the maskfor invalid regions
            original_image = torch.from_numpy(np.load(original_image_name)).float()
            if original_image.ndim == 2:
                original_image = original_image.unsqueeze(0)
            
            original_image = self.resize(original_image)
            original_mask = (original_image[0] == 0)
            original_image[0] = normalize_nonzero(original_image[0])
            
            mask = original_mask

        if os.path.exists(label_name):
            label_np = np.load(label_name).astype(np.int32) > 0
        else:
            label_np = np.zeros((self.img_size, self.img_size), dtype=np.int32)

        label = torch.from_numpy(label_np).unsqueeze(0).unsqueeze(0)
        label = torch.nn.functional.interpolate(label.float(), size=(self.resize_dim, self.resize_dim), mode='nearest').squeeze(0).squeeze(0).long()

        # Temporarily turn -1 → 255 for Albumentations
        label[label == -1] = 255

        if self.using_hillshade:
            hillshade_file_name, _ = self.expanded_file_list[idx]
            hillshade_path = os.path.join(self.root_dir, "hillshade", hillshade_file_name)
            hillshade = torch.from_numpy(np.load(hillshade_path)).float().unsqueeze(0)
            hillshade = self.resize(hillshade) / 255.0
            image = torch.cat([image, hillshade], dim=0)

        image_npy = image.permute(1, 2, 0).numpy().astype(np.float32)
        mask_npy = (mask.numpy() * 255).astype(np.int32)
        label_npy = label.numpy().astype(np.int32)
        masks = [mask_npy, label_npy]

        if self.transform:
            transformed = self.transform(image=image_npy, masks=masks)
            image_npy = transformed["image"]
            masks = transformed["masks"]

        image = torch.tensor(image_npy).permute(2, 0, 1).float()
        transformed_mask = torch.tensor(masks[0], dtype=torch.long)
        label = torch.tensor(masks[1], dtype=torch.long)

        label[label == 255] = -1
        label[transformed_mask == 255] = -1

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
