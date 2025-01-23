import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow_model_optimization.quantization.keras import vitis_quantize

# Define the necessary paths and parameters directly
model_path = 'resnet_50.h5'  # Path to the pre-trained ResNet50 model
quantize_output_dir = './quantized/'  # Directory to save the quantized model
eval_image_path = 'Imagenet_calib/Imagenet/val_dataset'  # Directory for evaluation images
eval_image_list = 'Imagenet_calib/Imagenet/val.txt'  # File containing the list of evaluation images
eval_batch_size = 10  # Batch size for evaluation

# Function to load image paths from a file
def get_images_infor_from_file(image_dir, image_list):

    with open(image_list, 'r') as fr:
        lines = fr.readlines()  # Read all lines in the file
    imgs = []
    for line in lines:
        img_name = line.strip().split(" ")[0]  # Extract the image name
        img_path = os.path.join(image_dir, img_name)  # Create full image path
        imgs.append(img_path)  # Append the image path to the list
    return imgs


# Define a custom Sequence class for ImageNet dataset
class ImagenetSequence(Sequence):
    
    def __init__(self, filenames, batch_size):
        self.filenames = filenames  # List of image file paths
        self.batch_size = batch_size  # Size of each batch

    def __len__(self):
        # Return the number of batches per epoch. """
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        # Load and preprocess a batch of images.
        # Get the filenames for the current batch
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        processed_imgs = []  # List to hold processed images

        for filename in batch_x:
            # Read and process the image
            img = cv2.imread(filename)  # Load the image using OpenCV
            height, width = img.shape[0], img.shape[1]  # Get the image dimensions
            img = img.astype(float)  # Convert image to float for processing

            # Aspect preserving resize
            smaller_dim = np.min([height, width])  # Find the smaller dimension
            _RESIZE_MIN = 256  # Minimum size for resizing
            scale_ratio = _RESIZE_MIN * 1.0 / (smaller_dim * 1.0)  # Calculate scale ratio
            new_height = int(height * scale_ratio)  # New height based on scale ratio
            new_width = int(width * scale_ratio)  # New width based on scale ratio
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)  # Resize the image

            # Central crop
            crop_height = 224  # Target crop height
            crop_width = 224  # Target crop width
            amount_to_be_cropped_h = (new_height - crop_height)  # Calculate amount to crop from height
            crop_top = amount_to_be_cropped_h // 2  # Top edge of crop
            amount_to_be_cropped_w = (new_width - crop_width)  # Calculate amount to crop from width
            crop_left = amount_to_be_cropped_w // 2  # Left edge of crop
            cropped_img = resized_img[crop_top:crop_top + crop_height,
                                       crop_left:crop_left + crop_width, :]  # Perform the crop

            # Subtract mean for normalization
            _R_MEAN = 123.68
            _G_MEAN = 116.78
            _B_MEAN = 103.94
            _CHANNEL_MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]  # Define means for each channel
            means = np.expand_dims(np.expand_dims(_CHANNEL_MEANS, 0), 0)  # Prepare means for subtraction
            meaned_img = cropped_img - means  # Subtract mean from the cropped image

            processed_imgs.append(meaned_img)  # Add processed image to the list

        return np.array(processed_imgs)  # Return batch of processed images as a NumPy array

# Function to quantize the model
def quantize_model():
    
    # Load the pre-trained ResNet50 model
    model = ResNet50(weights=model_path)

    # Load the evaluation images from the specified path
    img_paths = get_images_infor_from_file(eval_image_path, eval_image_list)

    # Prepare the image dataset for quantization using the first 100 images
    imagenet_seq = ImagenetSequence(img_paths[0:100], eval_batch_size)

    # Perform model quantization using the prepared dataset
    quantized_model = vitis_quantize.VitisQuantizer(model).quantize_model(calib_dataset=imagenet_seq)

    # Save the quantized model to the specified directory
    if not os.path.exists(quantize_output_dir):
        os.makedirs(quantize_output_dir)  # Create output directory if it doesn't exist
    quantized_model.save(os.path.join(quantize_output_dir, 'quantized.h5'))  # Save the quantized model

if __name__ == "__main__":
    quantize_model()  # Run the quantization process when the script is executed
