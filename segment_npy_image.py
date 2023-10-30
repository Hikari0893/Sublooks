import numpy as np
import matplotlib.pyplot as plt
from custom_bytescale import custom_bytescale
from rebin import rebin


def segment_npy_image(file_name, x_parts, y_parts, target_shape=None):
    # Load the npy file
    img_array = np.load(file_name)
    
    # Check if it's a 2-channel image
    if len(img_array.shape) == 3 and img_array.shape[-1] == 2:
        img_array = img_array[:, :, 0]

    # Determine segment dimensions
    height, width = img_array.shape
    segment_width = width // x_parts
    segment_height = height // y_parts

    segments = []

    # Create a single figure for all subplots
    fig, axes = plt.subplots(y_parts, x_parts, figsize=(15, 15))

    for i in range(x_parts):
        for j in range(y_parts):
            # Calculate start and end points for segments
            x_start, x_end = i * segment_width, (i + 1) * segment_width
            y_start, y_end = j * segment_height, (j + 1) * segment_height

            # Boundary checks to avoid out of range
            x_end = min(x_end, width)
            y_end = min(y_end, height)

            segment = img_array[y_start:y_end, x_start:x_end]

            # Check if segment is empty
            if segment.size == 0:
                print(f"Segment {i}_{j} is empty.")
                continue

            # Convert complex values to absolute values
            segment_abs = np.abs(segment)

            if target_shape:
                segment_abs = custom_bytescale(segment_abs)
                segment_abs = rebin(segment_abs, target_shape)
            
            segments.append(segment_abs)

            # Display on the corresponding subplot
            axes[j, i].imshow(segment_abs, cmap='gray')
            axes[j, i].set_title(f"Segment {i}_{j}")
            axes[j, i].axis("off")

    plt.tight_layout()
    plt.savefig("9_Segmentos.png")
    plt.show()

    return segments


#segment_npy_image("converted_image.npy", 3, 3, target_shape=(500,500))