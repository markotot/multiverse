import numpy as np
import math
import matplotlib.pyplot as plt

def pad_images(images: np.ndarray, top=0, bottom=0, left=0, right=0, constant=0) -> np.ndarray:
    assert len(images.shape) == 4, "not a batch of images!"
    return np.pad(images, ((0, 0), (top, bottom), (left, right), (0, 0)), mode="constant", constant_values=constant)

def plot_images(images, title, transpose):

    num_steps = len(images)
    if transpose:
        images = [np.transpose(obs, (1, 2, 0)) for obs in images]

    empty = np.array(images[0].copy())
    empty.fill(0)

    cols = math.sqrt(num_steps)
    if math.floor(cols) < cols:
        cols = math.floor(cols) + 1
    else:
        cols = math.floor(cols)  # for some reason this is needed

    rows = math.ceil(num_steps / cols)

    images.extend(((cols * rows) - len(images)) * [empty])

    padded_images = pad_images(np.array(images), top=4, bottom=4, left=4, right=4)
    image_rows = []
    resize_factor = 1
    for i in range(rows):
        image_slice = padded_images[i * cols: (i + 1) * cols]
        image_row = np.concatenate(image_slice, 1)
        x, y, _ = image_row.shape
        image_row_resized = image_row[::resize_factor, ::resize_factor]
        image_rows.append(image_row_resized)

    image = np.concatenate(image_rows, 0)

    plt.figure(dpi=300)
    plt.imshow(image)
    plt.axis('off')  # Optional: Turn off the axis
    plt.title(title)
    plt.show()


def plot_images_grid(images, figsize=(12, 6)):
    """
    Plot grayscale images in a 2x4 grid

    Args:
        images: numpy array of shape [batch_size, height, width, 1]
        figsize: tuple for figure size (width, height)
    """
    # Ensure we have at least 8 images for the 2x4 grid
    num_images = min(8, images.shape[0])

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()  # Flatten to easily iterate

    for i in range(num_images):
        # Remove the channel dimension for plotting (squeeze the last dimension)
        img = images[i].squeeze()

        axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(f'Image {i + 1}')
        axes[i].axis('off')  # Remove axis ticks and labels

    # Hide any unused subplots
    for i in range(num_images, 8):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()



# If you want to normalize pixel values (in case they're not in [0,1] range):
def plot_images_grid(images, figsize=(12, 6)):
    """
    Plot grayscale images in a 2x4 grid

    Args:
        images: numpy array of shape [batch_size, height, width, 1]
        figsize: tuple for figure size (width, height)
    """
    # Ensure we have at least 8 images for the 2x4 grid
    num_images = min(8, images.shape[0])

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()  # Flatten to easily iterate

    for i in range(num_images):
        # Remove the channel dimension for plotting (squeeze the last dimension)
        img = images[i].squeeze()

        axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(f'Image {i + 1}')
        axes[i].axis('off')  # Remove axis ticks and labels

    # Hide any unused subplots
    for i in range(num_images, 8):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()