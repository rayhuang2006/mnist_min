# Minimal CPU image for MNIST project with Gradio
FROM python:3.10-slim

# Avoid interactive tzdata etc.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (some wheels need these; PIL/OpenCV/torch vision often use libgl/libglib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
# Base requirements
RUN pip install --no-cache-dir -r requirements.txt
# Optional exporters/runtimes
RUN pip install --no-cache-dir onnx onnxruntime onnx2tf tensorflow-cpu==2.15.0

# Copy project
COPY . .

# Gradio port
EXPOSE 7860

# Default command runs the web UI
CMD ["python", "app.py", "--server.port", "7860", "--server.host", "0.0.0.0"]
