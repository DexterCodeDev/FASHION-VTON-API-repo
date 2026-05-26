# Use official PyTorch image with CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download weights into the image
# This assumes HF_TOKEN is available as a build-arg or cached
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='fashn-ai/fashn-vton-1.5', local_dir='./weights')"

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
