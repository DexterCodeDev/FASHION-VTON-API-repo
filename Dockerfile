# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies (Git is required to install from GitHub, libgl for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# PRE-DOWNLOAD WEIGHTS: We do this during the Docker build phase. 
# This bakes the 2GB model directly into the image so the cloud runner 
# doesn't time out downloading it every time a new instance spins up.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='fashn-ai/fashn-vton-1.5', local_dir='./weights')"

# Copy the rest of the application files
COPY . .

# Cloud Runners typically route traffic to port 8080
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
