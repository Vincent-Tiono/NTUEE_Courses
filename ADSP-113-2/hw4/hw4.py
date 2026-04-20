# Import required libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def convert_to_yuv420(img_file):
    """
    Convert an image to YUV 4:2:0 chroma subsampling format and back
    
    Args:
        img_file: Path to the input image
        
    Returns:
        Tuple containing (source_img, converted_img)
    """
    # Acquire image data (default OpenCV format is BGR)
    src_img = cv2.imread(img_file)
    if src_img is None:
        raise FileNotFoundError(f"Unable to load image from: {img_file}")
    
    # Transform to RGB color space
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    
    # Extract color channels
    red_channel = src_img[:, :, 0]
    green_channel = src_img[:, :, 1]
    blue_channel = src_img[:, :, 2]
    
    # Calculate YUV components using standard coefficients
    luma = 0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel
    chroma_red = (red_channel - luma) * 0.713 + 128
    chroma_blue = (blue_channel - luma) * 0.564 + 128

    # Ensure values are within valid range and convert to appropriate data type
    luma = np.clip(luma, 0, 255).astype(np.uint8)
    chroma_red = np.clip(chroma_red, 0, 255).astype(np.uint8)
    chroma_blue = np.clip(chroma_blue, 0, 255).astype(np.uint8)
    
    # Perform 4:2:0 subsampling (quarter resolution for chroma channels)
    chroma_red_subsampled = chroma_red[::2, ::2]
    chroma_blue_subsampled = chroma_blue[::2, ::2] 
    # Debug info (commented out):
    # print(f"Original chroma_red dimensions: {chroma_red.shape}, Subsampled: {chroma_red_subsampled.shape}")
    # print(f"Original chroma_blue dimensions: {chroma_blue.shape}, Subsampled: {chroma_blue_subsampled.shape}")

    # Restore chroma channels to original dimensions using bilinear interpolation
    chroma_red_upscaled = cv2.resize(
        chroma_red_subsampled, 
        (chroma_red.shape[1], chroma_red.shape[0]), 
        interpolation=cv2.INTER_LINEAR
    ).astype(np.uint8)
    
    chroma_blue_upscaled = cv2.resize(
        chroma_blue_subsampled, 
        (chroma_blue.shape[1], chroma_blue.shape[0]), 
        interpolation=cv2.INTER_LINEAR
    ).astype(np.uint8)

    # Combine channels into YCrCb format
    yuv_img = cv2.merge([luma, chroma_red_upscaled, chroma_blue_upscaled])

    # Transform back to RGB color space
    result_img = cv2.cvtColor(yuv_img, cv2.COLOR_YCrCb2RGB)

    return src_img, result_img

def visualize_comparison(source, processed):
    """
    Display source and processed images side by side with enhanced visualization
    
    Args:
        source: Original image
        processed: Processed image
    """
    # Configure visualization with a clean style
    plt.style.use('default')
    
    # Create figure with custom configuration
    fig = plt.figure(figsize=(14, 7))
    
    # Create GridSpec for more control over layout
    gs = fig.add_gridspec(2, 2, height_ratios=[6, 1], hspace=0.1)
    
    # Add subplots for images
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Display original image with enhanced appearance
    ax1.imshow(source, interpolation='nearest')
    ax1.set_title('Source Image', fontsize=14, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Display processed image with enhanced appearance
    ax2.imshow(processed, interpolation='nearest')
    ax2.set_title('4:2:0 Processed', fontsize=14, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Calculate PSNR
    psnr_value = compute_psnr(source, processed)
    
    # Add PSNR information in bottom area
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis('off')
    ax_info.text(0.5, 0.5, f"Image Quality (PSNR): {psnr_value:.2f} dB", 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox={'facecolor': 'lightgray', 'alpha': 0.7, 'pad': 10, 
                      'boxstyle': 'round,pad=0.5'})
    
    plt.show()

def compute_psnr(reference, test):
    """
    Calculate Peak Signal-to-Noise Ratio between two images
    Handles both grayscale and color images correctly
    
    Args:
        reference: Reference image (grayscale or color)
        test: Test image (grayscale or color)
        
    Returns:
        PSNR value in decibels (dB)
    """
    reference = reference.astype(np.float32)
    test = test.astype(np.float32)
    
    # Get image dimensions
    if len(reference.shape) == 3:  # Color image
        M, N, channels = reference.shape
        
        # Calculate MSE across all color channels
        total_squared_error = 0
        for c in range(channels):
            channel_error = np.sum((reference[:, :, c] - test[:, :, c]) ** 2)
            total_squared_error += channel_error
        
        # Average MSE across all pixels and channels (as per your formula)
        mean_squared_error = total_squared_error / (M * N * channels)
        
    else:  # Grayscale image
        M, N = reference.shape
        mean_squared_error = np.mean((reference - test) ** 2)
    
    # Handle perfect match case
    if mean_squared_error == 0:
        return float('inf')
    
    # PSNR calculation using the formula from your image
    max_pixel_value = 255.0
    psnr_value = 10 * math.log10((max_pixel_value ** 2) / mean_squared_error)
    
    return psnr_value

# Main execution
original, processed = convert_to_yuv420("dog.png")
visualize_comparison(original, processed)
print("Image Quality (PSNR):", compute_psnr(original, processed))
