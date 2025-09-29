import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
import re
import os
import requests
from typing import Optional, Tuple
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from pathlib import Path
##############################
# Google drive related utils #
##############################
def _extract_drive_file_id(drive_url: str) -> str:
    patterns = [
        r"drive\.google\.com\/file\/d\/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com\/open\?id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com\/uc\?id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com\/uc\?export=download&id=([a-zA-Z0-9_-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, drive_url)
        if m:
            return m.group(1)
    plain_id = re.fullmatch(r"[a-zA-Z0-9_-]{20,}", drive_url.strip())
    if plain_id:
        return drive_url.strip()
    raise ValueError("Could not parse a Google Drive FILE ID from the provided URL.")

def _get_confirm_token_from_html(text: str) -> Optional[str]:
    m = re.search(r'confirm=([0-9A-Za-z_-]+)', text)
    return m.group(1) if m else None

def _stream_to_file(response: requests.Response, destination_path: str, chunk_bytes: int = 1 << 20):
    os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
    total_bytes = int(response.headers.get("Content-Length", "0")) or None
    written = 0
    with open(destination_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_bytes):
            if chunk:
                f.write(chunk)
                written += len(chunk)
    if os.path.getsize(destination_path) == 0:
        raise IOError("Downloaded file is empty; the link may not be public or is invalid.")

def download_public_gdrive_file(drive_url_or_id: str, destination_path: str, timeout_sec: int = 60):
    file_id = _extract_drive_file_id(drive_url_or_id)
    base = "https://drive.google.com/uc"
    session = requests.Session()

    params = {"id": file_id, "export": "download"}
    r = session.get(base, params=params, stream=True, timeout=timeout_sec)
    r.raise_for_status()

    if "text/html" in r.headers.get("Content-Type", ""):
        token = _get_confirm_token_from_html(r.text)
        if not token:
            token = next((v for k, v in r.cookies.items() if k.startswith("download_warning")), None)

        if token:
            params["confirm"] = token
            r = session.get(base, params=params, stream=True, timeout=timeout_sec)
            r.raise_for_status()
        else:
            raise PermissionError(
                "Could not obtain confirm token. Ensure the file is PUBLIC or use Drive mounting."
            )

    _stream_to_file(r, destination_path)

#############################
# Visualization and logging #
#############################

@torch.no_grad()
def dump_visuals(loader: DataLoader, model, out_dir: Path,
                      max_items: Optional[int] = None, device: str = "cpu") -> None:
    """
    Save a single figure per batch with rows = batch size and columns = [Image | Label | Pred].
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)   # [B,3,H,W] or [B,1,H,W]
        labels = batch["label"].to(device)   # [B,H,W] or [B,1,H,W]
        if labels.dim() == 4:
            labels = labels.squeeze(1)

        logits = model(images)
        preds = logits.argmax(dim=1)         # [B,H,W]

        # how many rows to show from this batch (respect max_items if set)
        batch_size = images.size(0)
        rows_to_show = batch_size
        if max_items is not None:
            remaining = max(0, max_items - saved)
            if remaining == 0:
                break
            rows_to_show = min(rows_to_show, remaining)

        # create one figure per batch
        fig, axes = plt.subplots(rows_to_show, 3, figsize=(12, 3.2 * rows_to_show), squeeze=False)
        # titles on first row
        titles = ["Image", "Label", "Prediction"]
        for c, t in enumerate(titles):
            axes[0, c].set_title(t, fontsize=11)

        for r in range(rows_to_show):
            img = images[r].detach().cpu()             # [C,H,W]
            if img.dim() == 3 and img.size(0) in (1, 3):
                img = img if img.size(0) == 3 else img.repeat(3, 1, 1)
                img = img.clamp(0, 1).permute(1, 2, 0).numpy()  # -> [H,W,3]
            else:
                img = img.squeeze().clamp(0, 1).numpy()
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)

            lab = labels[r].detach().cpu()
            if lab.max() > 1:
                lab = (lab > 127).to(lab.dtype)
            lab_np = lab.numpy()

            pred_np = preds[r].detach().cpu().numpy()

            axes[r, 0].imshow(img)
            axes[r, 0].axis("off")

            axes[r, 1].imshow(lab_np, cmap="gray", vmin=0, vmax=1)
            axes[r, 1].axis("off")

            axes[r, 2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            axes[r, 2].axis("off")

        plt.tight_layout()
        fig_path = out_dir / f"batch_{batch_idx:06d}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)

        saved += rows_to_show
        if (max_items is not None) and (saved >= max_items):
            break


@torch.no_grad()
def compute_balanced_weights(loader, ignore_index= -1) -> Tuple[float, float]:
    # anja
    fg_count = 0
    bg_count = 0
    for batch in loader:
        labels = batch["label"]
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        valid_mask = (labels != ignore_index)
        fg_count += (labels.eq(1) & valid_mask).sum().item()
        bg_count += (labels.eq(0) & valid_mask).sum().item()
    eps = 1e-6
    total_valid = max(bg_count, 1)
    ratio = fg_count / (total_valid + eps)
    weight1 = fg_count / (total_valid - fg_count + eps)
    weight0 = 1.0 / (weight1 + eps)
    print(f"Label ratio (fg/valid): {ratio:.6f} | Weight0: {weight0:.6f} | Weight1: {weight1:.6f}")
    return float(weight0), float(weight1)

@torch.no_grad()
def foreground_iou_from_logits(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -1) -> float:
    if labels.dim() == 4:
        labels = labels.squeeze(1)
    valid = (labels != ignore_index)
    if valid.sum() == 0:
        return 1.0
    preds = logits.argmax(dim=1)
    preds = preds[valid]
    gt    = labels[valid]

    inter = ((preds == 1) & (gt == 1)).sum().float()
    union = ((preds == 1) | (gt == 1)).sum().float()
    if union == 0:
        return 1.0
    return (inter / union).item()

def visualize_triplets_inline(trained_model, loader, device, num_items: int = 3):
    trained_model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)                   
            labels = batch["label"]                             
            if labels.dim() == 4:
                labels = labels.squeeze(1)
            logits = trained_model(images)
            preds  = logits.argmax(dim=1).cpu()                 

            images_np = images.detach().cpu().clamp(0,1).permute(0,2,3,1).numpy()
            labels_np = labels.detach().cpu().numpy()
            preds_np  = preds.detach().cpu().numpy()

            k = min(num_items, images_np.shape[0])
            fig, axes = plt.subplots(k, 3, figsize=(9, 3*k))
            if k == 1:
                axes = np.expand_dims(axes, axis=0)

            for i in range(k):
                axes[i, 0].imshow(images_np[i])
                axes[i, 0].set_title("Image")
                axes[i, 0].axis("off")

                axes[i, 1].imshow(labels_np[i], cmap="gray", vmin=0, vmax=1)
                axes[i, 1].set_title("Label")
                axes[i, 1].axis("off")

                axes[i, 2].imshow(preds_np[i], cmap="gray", vmin=0, vmax=1)
                axes[i, 2].set_title("Prediction")
                axes[i, 2].axis("off")

            plt.tight_layout()
            plt.show()


def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def save_combined_image(image, pred, label, test_file, pred_path, ship_iou=None, terrain_iou=None, divider_width=5, saliency_map=None):
    img = image.squeeze().cpu().numpy()
    img = (255 * img).astype(np.uint8)
    img_pil = Image.fromarray(img)

    pred_img = Image.fromarray((127.5 * pred[0, ...]+127.5).astype(np.uint8))
    label_img = Image.fromarray((127.5*label.squeeze()+127.5).astype(np.uint8))

    height = img_pil.height
    pred_img = pred_img.resize((pred_img.width, height))
    label_img = label_img.resize((label_img.width, height))
    saliency_img = None
    
    if saliency_map is not None:
        saliency_img = Image.fromarray(255 * saliency_map[0])
        saliency_img = saliency_img.resize((img_pil.width, height))

    divider = Image.new("RGB", (divider_width, height), color=(128, 128, 128))

    combined_width = img_pil.width + pred_img.width + label_img.width + (2 * divider_width)
    if saliency_map is not None:
        combined_width += saliency_img.width + divider_width
    combined_img = Image.new("RGB", (combined_width, height), color=(0, 0, 0))

    x_offset = 0
    combined_img.paste(img_pil, (x_offset, 0))
    x_offset += img_pil.width
    combined_img.paste(divider, (x_offset, 0))
    x_offset += divider_width
    combined_img.paste(label_img.convert("RGB"), (x_offset, 0))
    x_offset += pred_img.width
    combined_img.paste(divider, (x_offset, 0))
    x_offset += divider_width
    combined_img.paste(pred_img.convert("RGB"), (x_offset, 0))
    if saliency_map is not None:
        x_offset += pred_img.width
        combined_img.paste(divider, (x_offset, 0))
        x_offset += divider_width
        combined_img.paste(saliency_img.convert("RGB"), (x_offset, 0))

    if ship_iou is not None or terrain_iou is not None:
        draw = ImageDraw.Draw(combined_img)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()

        text = f"Ship IoU: {ship_iou:.4f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (combined_img.width - text_width) // 2
        text_y = combined_img.height - text_height - 10

        padding = 5
        draw.rectangle(
            [text_x - padding, text_y - padding, text_x + text_width + padding, text_y + text_height + padding],
            fill=(255, 255, 255)
        )

        draw.text((text_x, text_y), text, font=font, fill=(255, 0, 0))  # Red text

    # Save the final combined image
    output_filename = os.path.basename(test_file).replace("_image.npy", "_combined.png")
    pred_img.save(os.path.join(pred_path, output_filename))
    
    return combined_img

def save_plot(save_path, data):
    # Create the figuredepth_grid
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='gray', interpolation='nearest')
    plt.colorbar()  # Optional: add a colorbar
    plt.axis('off')  # Optional: remove axes for a cleaner look

    # Save the figure
    plt.savefig(save_path)

    # Show the figure (optional)
    # plt.show()

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

def copy_files(src_folder, dst_folder):
    """
    Copies all files from src_folder to dst_folder.
    
    Args:
        src_folder (str): Source directory path.
        dst_folder (str): Destination directory path.
    """
    # Ensure the destination folder exists
    os.makedirs(dst_folder, exist_ok=True)

    # Loop through all files in the source directory
    for count, filename in enumerate(os.listdir(src_folder)):
        src_file = os.path.join(src_folder, filename)
        dst_file = os.path.join(dst_folder, filename)

        # Check if it's a file before copying
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)  # copy2 preserves metadata

    print(f"Copied {count} ship images to {dst_folder}.")

    # print(f"All files copied from {src_folder} to {dst_folder}")
