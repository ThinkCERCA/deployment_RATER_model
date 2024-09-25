
# PyTorch GPU Model Deployment on AWS using Docker

This project demonstrates how to deploy a PyTorch model on AWS using Docker with GPU acceleration. The model is deployed in a Docker container with PyTorch and CUDA support for GPU-based computation.

## Prerequisites

Before you begin, ensure you have the following:

- An AWS EC2 instance with GPU support (e.g., a `g4dn.xlarge` instance).
- Docker installed on the instance.
- NVIDIA drivers and CUDA toolkit installed for GPU support.

## Setup Instructions

## Connect to EC2 and Upload Model

### Step 1: Connect to the EC2 Instance with PuTTY

To access your AWS EC2 instance, follow these steps:

1. **Download PuTTY**: Download and install [PuTTY](https://www.putty.org/), an SSH client.
2. **Convert your PEM key to PPK**: Use PuTTYgen to convert your AWS PEM file to a PPK format.
   - Open **PuTTYgen**.
   - Click **Load** and choose your `.pem` file.
   - Click **Save private key** to save a `.ppk` file.
3. **Connect to EC2**:
   - Open **PuTTY**.
   - In the **Host Name (or IP address)** field, enter `ec2-user@<your-ec2-public-ip>`.
   - In **Category > Connection > SSH > Auth**, browse for your `.ppk` file under **Private key file for authentication**.
   - Click **Open** to start the SSH session.
   - user name is `ec2-user` or `ubuntu`

### Step 2: Upload the Model to EC2 with WinSCP

1. **Download WinSCP**: Install [WinSCP](https://winscp.net/eng/index.php), a file transfer tool.
2. **Connect to EC2**:
   - Open **WinSCP**.
   - In the **Host Name** field, enter the EC2 instance's public IP address.
   - Set the **Username** to `ec2-user` or `ubuntu`.
   - Select your **Private Key File** (the `.ppk` file generated earlier).
   - Click **Login** to connect.
3. **Upload the Model**: 
   - Once connected, navigate to the desired directory on the EC2 instance.
   - Upload your model file from your local machine to the instance by dragging and dropping it into the remote directory.



### Step 3: Install NVIDIA Drivers

Run the following commands to install the NVIDIA drivers and the necessary container toolkit:

```bash
sudo ubuntu-drivers autoinstall
sudo apt install nvidia-cuda-toolkit -y
```

### Step 4: Install NVIDIA Container Toolkit

Add the NVIDIA Container Toolkit repository and install the toolkit using these commands:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo reboot
```

### Step 5: Pull the PyTorch Docker Image

After the instance reboots, pull the PyTorch Docker image with CUDA support:

```bash
docker pull pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
```

### Step 6: Run the Docker Container

To run the PyTorch container with GPU support and map port 8080 for the API:

```bash
sudo docker run --gpus all -it --rm -v $PWD:/workspace -p 8080:8080 pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
```

This command launches the Docker container with access to all GPUs, and maps your local directory to `/workspace` in the container.

### Step 7: Copy the Model from EC2 to the Docker Container

After uploading the model to the EC2 instance, you can copy it into your running Docker container:

1. **Find the Container ID**:
   Run the following command to find your container's ID:

   ```bash
   docker ps
   ```

2. **Copy the Model into the Container**:
   Use the `docker cp` command to copy your model file from the EC2 instance to the container:

   ```bash
   docker cp /path/to/model.pt <container_id>:/workspace
   ```

   Replace `<container_id>` with your actual container ID and adjust `/workspace` as per your container setup.

3. **Verify the Model in the Container**:
   To check if the file has been successfully copied, open a bash session inside the container:

   ```bash
   docker exec -it <container_id> bash
   ```

   Then list the files in the `/workspace` directory to confirm the presence of your model:

   ```bash
   ls /workspace
   ```

Now, your model is ready for use in the container.

### Step 8: Run the Model API

Inside the container, run your model script:

```bash
nohup python run.py &
```

You can check that the Python process is running:

```bash
ps aux | grep python
```

### Step 9: Test the API

Once the model is running in the container, you can test it using `curl` or Postman. For example, send a POST request with a JSON payload:

```bash
curl -X POST -H "Content-Type: application/json" -d '{ 
  "content": "The biggest threat to the everglades is the draining of the swamp this is the biggest threat because in the begging of the artical all it talk about was how this cause so much problem for the people that lived their. A reason why this was so much of a problem was because ''according to the text for centuries , however humans thought of wetlands as unhygienic swamps. they didnt even give it a chance they just thought it was nasty  Another reason why this is a problem is because much of the northern iarea is polluted ''according to the text  much of the northern are has been polluted with with phosphrous."
}' http://3.238.237.26:8080/
```

This request sends a sample text to the deployed model and returns the rater results in about 3 seconds.




