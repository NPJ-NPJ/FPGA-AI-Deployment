import os
import numpy as np
import xir
import vart
from PIL import Image  # Import Pillow for image processing

# Class to hold information about input and output tensors
class GraphInfo:
    def __init__(self):
        self.inTensorList = []  # List to store input tensors
        self.outTensorList = []  # List to store output tensors

# Define paths for images and class labels
image_path = "./images/"  # Directory where input images are stored
class_path = "./words.txt"  # File containing class names for the model output
model_file = "./compiled/Compiled_model.xmodel"  # Path to the compiled model file

# Function to load class names(labels) from a file
def load_words(path): 
    with open(path, 'r') as f:
        return f.read().splitlines()  # Read class names into a list

# Function to calculate softmax probabilities
def calc_softmax(data):
    exp_data = np.exp(data - np.max(data))  # Apply exponentiation for numerical stability
    return exp_data / exp_data.sum()  # Normalize the exponentiated values to get probabilities


# Function to display the top k predictions
def top_k(probabilities, k, kinds):
    indices = np.argsort(probabilities)[-k:][::-1]  # Get indices of the top k probabilities
    for i in indices:
        print(f"top[{len(indices) - 1 - np.where(indices == i)[0][0]}] prob = {probabilities[i]:.8f} name = {kinds[i]}")

# Load class names for model output
kinds = load_words(class_path)  # Load class names from the specified file
num_classes = len(kinds)  # Determine the number of class labels

# Load the model graph and prepare to run inference
graph = xir.Graph.deserialize(model_file)  # Load the model graph from the compiled model file
subgraph = graph.get_root_subgraph().toposort_child_subgraph()  # Retrieve the subgraph for inference

dpu1 = vart.Runner.create_runner(subgraph[1], "run")  # Create a runner for executing the model

# Define mean values for image preprocessing
mean = np.array([104, 107, 123], dtype=np.float32)  # Mean values for each color channel (R, G, B)

# Retrieve output and input tensors from the runner
output_tensors = dpu1.get_output_tensors()  # Get output tensors from the model
input_tensors = dpu1.get_input_tensors()  # Get input tensors required by the model

# Get the dimensions of the input and output tensors
out_dims = tuple(output_tensors[0].dims)  # Shape of the output tensor
in_dims = tuple(input_tensors[0].dims)  # Shape of the input tensor

# Prepare an array to hold the output data
out1 = np.zeros(out_dims, dtype='float32')  # Create an empty array for output results

# Prepare an array for the input image data
image_inputs = np.zeros((1, in_dims[1], in_dims[2], 3), dtype=np.int8)  # Shape (1, height, width, channels)

# Load and preprocess the image
image = Image.open(os.path.join(image_path, "001.jpg"))  # Open the image file
image = image.resize((in_dims[2], in_dims[1]))  # Resize the image to match the model input dimensions
image = np.array(image)  # Convert the image to a NumPy array

# Preprocess the image data by subtracting the mean values
for i in range(3):  # Loop through the three color channels (R, G, B)
    image_inputs[0, :, :, i] = (image[:, :, i] - mean[i]).astype(np.int8)  # Normalize the color channel

inputData = image_inputs  # Assign processed image data to inputData
# Execute the model asynchronously with the input data and output buffer
job_id = dpu1.execute_async([inputData], [out1])  # Execute the model
dpu1.wait(job_id)  # Wait for the execution to complete

# Calculate softmax probabilities from the output data
softmax_probs = calc_softmax(out1.squeeze())  # Apply softmax to get probability distribution, squeeze is used to remove extra dimensions
#print(softmax_probs)  # Print the softmax probabilities

# Display the top 5 results based on the calculated probabilities
top_k(softmax_probs, 5, kinds)  # Show top 5 predictions with their probabilities
