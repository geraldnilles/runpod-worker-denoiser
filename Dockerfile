
# Use verified Ubuntu 24.04 base
FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install Vulkan tools
# vulkan-tools: contains vulkaninfo
# libvulkan1: the loader required to interface with the GPU driver
RUN apt-get update && apt-get install -y \
    vulkan-tools \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# CRITICAL: This environment variable tells the Nvidia Container Toolkit
# to inject the 'graphics' driver libraries (needed for Vulkan)
# alongside the 'compute' libraries (needed for CUDA).
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

# Standard variable to ensure the GPU is visible
ENV NVIDIA_VISIBLE_DEVICES=all

# default command to verify setup
CMD ["vulkaninfo","--summary"]


