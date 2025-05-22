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


