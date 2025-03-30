FROM nvidia/cuda:12.4.0-cudnn8-devel-ubuntu22.04  # Base CUDA image

WORKDIR /app
COPY . /app/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["/bin/bash"]