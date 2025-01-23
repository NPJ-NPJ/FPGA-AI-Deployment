
# import the libraries
import tensorflow as tf 
from tensorflow_model_optimization.quantization.keras import vitis_quantize

# Load the float model
model = tf.keras.models.load_model('resnet_50.h5')

# Import Vitis Inspector
from tensorflow_model_optimization.quantization.keras import vitis_inspect
inspector = vitis_inspect.VitisInspector(target="/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json")     # Specify the target device DPU

# Inspect the model and generate inspection results
inspector.inspect_model(model, 
                        plot=True,  # Enable plot generation
                        plot_file="resnet_50_model.svg",  # Output file for the plot
                        dump_results=True,  # Enable dumping results
                        dump_results_file="resnet_50_model_inspect_results.txt",  # File to save dumped results
                        verbose=0)  # Set verbosity level - 0: no output to stdout, 1: output to stdout
