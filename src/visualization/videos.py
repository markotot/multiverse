import cv2
import numpy as np

def create_video_from_images(images, episode_returns, output_path, fps, scale_factor):
    """
    Create a video from a numpy array of grayscale images with frame number overlay.

    Args:
        images: numpy array of shape (n_frames, 84, 84) containing grayscale images
        output_path: path for the output video file
        fps: frames per second for the output video
        scale_factor: factor to scale up the small 84x84 images for better visibility
    """

    if len(images.shape) != 3 or images.shape[1:] != (84, 84):
        raise ValueError("Images should be a 3D array with shape (n_frames, 84, 84)")

    n_frames, height, width = images.shape

    # Scale up the images for better visibility
    scaled_height = height * scale_factor
    scaled_width = width * scale_factor

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (scaled_width, scaled_height))

    for i, img in enumerate(images):
        # Normalize image to 0-255 range if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        # Scale up the image
        scaled_img = cv2.resize(img, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)

        # Convert grayscale to BGR for video
        bgr_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2BGR)

        # Add frame number overlay
        frame_text = f"Score: {episode_returns[i]}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255)  # White text

        # Get text size for positioning
        (text_width, text_height), baseline = cv2.getTextSize(frame_text, font, font_scale, font_thickness)

        # Position text at top-left with some padding
        text_x = 190
        text_y = text_height + 10

        # Add a black background rectangle for better text visibility
        cv2.rectangle(bgr_img,
                      (text_x - 5, text_y - text_height - 5),
                      (text_x + text_width + 5, text_y + baseline + 5),
                      (0, 0, 0), -1)

        # Add the text
        cv2.putText(bgr_img, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

        # Write the frame
        out.write(bgr_img)

    # Release everything
    out.release()
    print(f"Video saved as {output_path}")


if __name__ == "__main__":
    # Create sample data (replace this with your actual image array)
    # Assuming you have your images in a variable called 'your_images'

    # Sample: create some dummy grayscale images
    n_frames = 50
    sample_images = np.random.rand(n_frames, 84, 84)  # Random grayscale images

    # Add some pattern to make it more interesting
    for i in range(n_frames):
        # Add a moving circle
        center_x = int(42 + 20 * np.sin(i * 0.2))
        center_y = int(42 + 20 * np.cos(i * 0.2))
        cv2.circle(sample_images[i], (center_x, center_y), 10, 1.0, -1)

    # Create video from your images
    # Replace 'sample_images' with your actual image array
    create_video_from_images(
        sample_images,  # Your numpy array of images
        output_path='../../my_video.mp4',
        fps=15,
        scale_factor=6  # Makes 84x84 -> 504x504 for better visibility
    )