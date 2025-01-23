# Accelerating AI Inference on Hardware (FPGAs)

## Overview

This repository provides resources for using **Vitis AI**, Xilinx's AI acceleration framework, and the **Deep Learning Processing Unit (DPU)** to efficiently run neural networks on FPGAs. The workflow demonstrates the deployment of a pre-trained ResNet-50 model using TensorFlow 2, optimization for FPGA deployment, and acceleration of deep learning inference.

### Key Workflow Stages:
- **Quantization**: Convert the floating-point model to an FPGA-compatible format, reducing size and improving performance.
- **Compilation**: Use the Vitis AI Compiler to optimize the model for the DPU architecture.
- **Inference**: Deploy the optimized model on FPGA hardware to achieve significant improvements in speed and energy efficiency.

---

## Repository Contents

The repository includes the following files and resources:

1. **`inspect_float_model.py`**
   - This script inspects the floating-point model to extract details such as:
     - Layers
     - Input shapes
     - Subgraphs
   - **Why Subgraphs Are Important**:  
     Subgraphs represent portions of the neural network that can be offloaded to the FPGA's DPU for hardware acceleration. The rest of the network executes on the CPU. This division helps maximize efficiency and performance.  
     

2. **`quantize_model.py`**
   - This script handles the quantization process, converting the floating-point ResNet-50 model into an 8-bit integer representation compatible with FPGA DPUs.  
   - **Input Images for Quantization**:  
     The quantization process requires a dataset of representative input images. These images can be downloaded from standard datasets such as [ImageNet](http://www.image-net.org/).

3. **`analyse_subgraphs.sh`**
   - This shell script analyzes the compiled model and retrieves detailed subgraph information.  
   - It helps verify which parts of the model are mapped to the DPU and provides insights for debugging or optimization.

4. **`inference_on_board.py`**
   - This script runs inference on the FPGA hardware using the compiled model (Xmodel).  
   - Ensure paths are correctly configured for the model and dataset before running.

5. **Model Files and Configuration**
   - Pre-trained ResNet-50 model (`float` and `quantized` versions).
   - Compiled Xmodel for deployment.

6. **Support Files**
   - Commands and configurations necessary to run Vitis AI workflows.
   - Example scripts for setting up the FPGA board and executing tasks.

---

## Instructions

### 1. Environment Setup
- Install **Vitis AI 3.5** and **Docker** (v19.03+).  



### 2. Launch Vitis AI Docker Container
- Use the following command to launch the tensorflow2 container:
  ```bash
  ./docker_run.sh xilinx/vitis-ai-tensorflow2-cpu:latest


### 3. Inspect the Float Model
```
python3 inspect_float_model.py
```

### 4.Quantise the model 
```
python3 quantize_model.py

```

### 5. Compile the model for the FPGA (Here the model is combiled for Kria KV260, DPU DPUCZDX8G)
```bash 
 vai_c_tensorflow2 -m quantized/quantized.h5 -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o ./compiled -n resnet50
```

## FPGA Setup and Inference Deployment Guide

### 4. Set Up the FPGA Board

1. **Flash the Vitis AI SD Card Image**
   - Download the pre-built Vitis AI SD card image for Kria KV260 from the official website. [Kria Kv260](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-kv260-dpu-v2022.2-v3.0.0.img.gz)

2. **Boot the Board**
   - Insert the flashed SD card into the Kria KV260 board.
   - Power on the board. 

3. **Establish Ethernet and Serial Connections**
   - Connect an Ethernet cable to the board to establish a network connection.
   - Use a serial terminal (e.g., PuTTY /TeraTerm /Picocom) to connect to the board via a UART interface for debugging.

## 5. Deploy and Run Inference
1. **Transfer the Compiled Model and Scripts to the FPGA**
   - Transfer the compiled model ,inference scripts and the data (images and the labels) to the Kria KV260 via SCP
     - Example command:
       ```bash
       scp -r ./compiled root@<ip_address_of_fpga>:/home/root/workspace/

       ```

2. **Run the Inference**
   - SSH into the Kria KV260 board and run the inference script:
     ```bash
     ssh user@<FPGA_IP>
     python3 inference_script.py
     ```

   - The inference result will be printed in the terminal. Ensure that the model and script are correctly configured


   ### References
   - [Docker Installation Guide](https://docs.docker.com/engine/install/ubuntu/).
   - [Vitis AI 3.5 Installation Guide](https://xilinx.github.io/Vitis-AI/3.5/html/docs/install/install.html)
   - [Kria KV260 Setup Guide](https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html)
