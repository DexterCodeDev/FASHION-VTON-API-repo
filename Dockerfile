FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HF_HOME=/app/hf_cache

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Clone VTON repo automatically
RUN git clone https://github.com/fashn-AI/fashn-vton-1.5.git

# Install VTON package
RUN pip install -e ./fashn-vton-1.5

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY start.sh .

RUN chmod +x start.sh

EXPOSE 8080

CMD ["bash", "start.sh"]
