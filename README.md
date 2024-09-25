
# PyTorch Model Deployment on AWS using Docker

This project demonstrates how to deploy a PyTorch model on AWS using Docker with GPU acceleration. The model is deployed in a Docker container with PyTorch and CUDA support for GPU-based computation.

## Prerequisites

Before you begin, ensure you have the following:

- An AWS EC2 instance with GPU support (e.g., a `g4dn.xlarge` instance).
- Docker installed on the instance.
- NVIDIA drivers and CUDA toolkit installed for GPU support.

## Setup Instructions

### Step 1: Install NVIDIA Drivers

Run the following commands to install the NVIDIA drivers and the necessary container toolkit:

\`\`\`bash
sudo ubuntu-drivers autoinstall
sudo apt install nvidia-cuda-toolkit -y
\`\`\`

### Step 2: Install NVIDIA Container Toolkit

Add the NVIDIA Container Toolkit repository and install the toolkit using these commands:

\`\`\`bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo reboot
\`\`\`

### Step 3: Pull the PyTorch Docker Image

After the instance reboots, pull the PyTorch Docker image with CUDA support:

\`\`\`bash
docker pull pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
\`\`\`

### Step 4: Run the Docker Container

To run the PyTorch container with GPU support and map port 8080 for the API:

\`\`\`bash
sudo docker run --gpus all -it --rm -v $PWD:/workspace -p 8080:8080 pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
\`\`\`

This command launches the Docker container with access to all GPUs, and maps your local directory to `/workspace` in the container.

### Step 5: Run the Model API

Inside the container, run your model script:

\`\`\`bash
nohup python run.py &
\`\`\`

You can check that the Python process is running:

\`\`\`bash
ps aux | grep python
\`\`\`

### Step 6: Test the API

Once the model is running in the container, you can test it using `curl` or Postman. For example, send a POST request with a JSON payload:

\`\`\`bash
curl -X POST -H "Content-Type: application/json" -d '{ 
  "content": "The biggest threat to the everglades is the draining of the swamp this is the biggest threat because in the begging of the artical all it talk about was how this cause so much problem for the people that lived their. A reason why this was so much of a problem was because ''according to the text for centuries , however humans thought of wetlands as unhygienic swamps. they didnt even give it a chance they just thought it was nasty  Another reason why this is a problem is because much of the northern iarea is polluted ''according to the text  much of the northern are has been polluted with with phosphrous."
}' http://3.238.237.26:8080/
\`\`\`

This request sends a sample text to the deployed model and returns the rater results in about 3 seconds.

## Accessing the API

To access the API from outside the network, you may need to connect to the ThinkCERCA VPN. Contact Bryant for assistance with VPN access.

## Future Enhancements

- Add request code to our demo, allowing direct integration of the model results into the system's prompt.
- Improve model optimization and reduce response time.

## Contact

For any questions or issues, feel free to reach out to the team.
