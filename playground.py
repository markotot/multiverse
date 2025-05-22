import torch
import torch.nn.functional as F


def rescale_grayscale_images(grayscale_images, target_size=(84, 84), mode='bilinear'):
    """
    Rescale grayscale images to a target size.

    Parameters:
    -----------
    grayscale_images : torch.Tensor
        Input tensor of shape [batch_size, height, width]
    target_size : tuple
        Target (height, width) to rescale to
    mode : str, optional
        Interpolation mode. Options:
        - 'nearest': nearest neighbor interpolation
        - 'bilinear': bilinear interpolation (default)
        - 'bicubic': bicubic interpolation

    Returns:
    --------
    torch.Tensor
        Rescaled images of shape [batch_size, target_height, target_width]
    """
    # Add channel dimension for interpolation
    x = grayscale_images.unsqueeze(1)

    # Rescale using interpolate
    rescaled = F.interpolate(x, size=target_size, mode=mode, align_corners=False)

    # Remove channel dimension
    return rescaled.squeeze(1)


# Example usage
if __name__ == '__main__':
    # Create a sample grayscale tensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 10 images, 64x64 pixels
    sample_grayscale = torch.randint(0, 256,
                                     size=(10, 64, 64),
                                     dtype=torch.float32,
                                     device=device)

    # Rescale using different interpolation modes
    print("Original shape:", sample_grayscale.shape)

    # Bilinear interpolation (default)
    bilinear_rescaled = rescale_grayscale_images(sample_grayscale, target_size=(84, 84), mode='bilinear')
    print("Bilinear rescaled shape:", bilinear_rescaled.shape)

    # Nearest neighbor interpolation
    nearest_rescaled = rescale_grayscale_images(sample_grayscale, target_size=(84, 84), mode='nearest')
    print("Nearest neighbor rescaled shape:", nearest_rescaled.shape)

    # Bicubic interpolation
    bicubic_rescaled = rescale_grayscale_images(sample_grayscale, target_size=(84, 84), mode='bicubic')
    print("Bicubic rescaled shape:", bicubic_rescaled.shape)