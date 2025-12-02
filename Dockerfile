
# Use verified Ubuntu 24.04 base
FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install Vulkan tools
RUN apt-get update && apt-get install -y \
    vulkan-tools \
    libvulkan1 \
    libx11-6 \
    libxext6 \
    libgl1 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/vulkan/icd.d && \
    echo '{ "file_format_version" : "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version" : "1.3" } }' > /etc/vulkan/icd.d/nvidia_icd.json


# CRITICAL: This environment variable tells the Nvidia Container Toolkit
# to inject the 'graphics' driver libraries (needed for Vulkan)
# alongside the 'compute' libraries (needed for CUDA).
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

# Standard variable to ensure the GPU is visible
ENV NVIDIA_VISIBLE_DEVICES=all

# default command to verify setup
CMD ["vulkaninfo","--summary"]


