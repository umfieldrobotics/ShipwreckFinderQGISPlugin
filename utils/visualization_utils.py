import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import re
from PIL import Image, ImageDraw, ImageFont
from IPython.display import Image as IPyImage, display


def make_labeled_segmentation_gif(
    target_dir: Path,
    frame_glob: str = "epoch_*/batch_000001.png",
    out_name: str = "segmentation_evolution.gif",
    duration_ms: int = 500,
):
    frame_paths = sorted(target_dir.glob(frame_glob))
    frames = []
    base_size = None
    font = ImageFont.load_default()
    font.size = 20

    def epoch_from_path(p: Path) -> int:
        m = re.search(r"epoch_(\d+)", str(p.parent.name))
        return int(m.group(1)) if m else -1

    for frame_path in frame_paths:
        if not frame_path.exists():
            continue
        im = Image.open(frame_path).convert("RGB")
        if base_size is None:
            base_size = im.size
        elif im.size != base_size:
            im = im.resize(base_size, Image.NEAREST)

        # ---- draw epoch label ----
        epoch_num = epoch_from_path(frame_path)
        draw = ImageDraw.Draw(im)
        text = f"Epoch {epoch_num:03d}" if epoch_num >= 0 else "Epoch ?"
        # background box for readability
        pad = 6
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        box = [5, 5, 5 + tw + 2 * pad, 5 + th + 2 * pad]
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text((5 + pad, 5 + pad), text, fill=(255, 255, 255), font=font)

        frames.append(im)

    if not frames:
        print("No frames found to build GIF.")
        return None

    out_path = target_dir / out_name
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return out_path


def to_numpy_image01(image_tensor_or_array):
    if isinstance(image_tensor_or_array, torch.Tensor):
        img = image_tensor_or_array.detach().cpu().numpy()
    else:
        img = np.asarray(image_tensor_or_array)

    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.clip(img, 0.0, 1.0)
    return img


def to_numpy_label01(label_tensor_or_array):
    if isinstance(label_tensor_or_array, torch.Tensor):
        lab = label_tensor_or_array.detach().cpu().numpy()
    else:
        lab = np.asarray(label_tensor_or_array)

    if lab.ndim == 3 and lab.shape[0] == 1:
        lab = lab[0]
    if lab.ndim == 3 and lab.shape[-1] == 1:
        lab = lab[..., 0]

    if lab.dtype != np.float32:
        lab = lab.astype(np.float32)
    if lab.max() > 1.0:
        lab = (lab > 127).astype(np.float32)
    return lab


def overlay_label_on_image01(image01_hwc, label01_hw, alpha=0.4):
    color_label = np.zeros((*label01_hw.shape, 3), dtype=np.float32)
    color_label[label01_hw == 1] = [1.0, 0.0, 0.0]  # red
    return (1 - alpha) * image01_hwc + alpha * color_label


def plot_train_test_grid(train_dataset, test_dataset, N=5):
    fig, axes = plt.subplots(N, 7, figsize=(14, 2*N))
    titles = [
        "Train Image",
        "Train Label",
        "Train Overlay",
        "",
        "Test Image",
        "Test Label",
        "Test Overlay",
    ]
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=11)

    for r in range(N):
        train_sample = train_dataset[r % len(train_dataset)]
        train_img = to_numpy_image01(train_sample["image"])
        train_lab = to_numpy_label01(train_sample["label"])
        train_ovr = overlay_label_on_image01(train_img, train_lab)

        axes[r, 0].imshow(train_img, cmap="gray")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(train_lab, cmap="gray", vmin=0, vmax=1)
        axes[r, 1].axis("off")

        axes[r, 2].imshow(train_ovr)
        axes[r, 2].axis("off")

        # leavae empty column
        axes[r, 3].axis("off")

        test_sample = test_dataset[r % len(test_dataset)]
        test_img = to_numpy_image01(test_sample["image"])
        test_lab = to_numpy_label01(test_sample["label"])
        test_ovr = overlay_label_on_image01(test_img, test_lab)

        axes[r, 4].imshow(test_img,cmap="gray")
        axes[r, 4].axis("off")

        axes[r, 5].imshow(test_lab, cmap="gray", vmin=0, vmax=1)
        axes[r, 5].axis("off")

        axes[r, 6].imshow(test_ovr)
        axes[r, 6].axis("off")

    plt.tight_layout()
    plt.show()
