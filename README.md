# Runpod Worker Denoiser

This repo will generate a Runpod serverless container that will accept an image as an input and return a denoised image. 

My personal goal is to have a quick way of using SOTA models for cleaning up the photos taken by my mirrorless camera. 

## Model

I will start by using the ESRGAN-based models, but I might experiment with other architectures as well if this goes well.

This is the model I have played around with on Colab and it seems to work pretty well.
<https://openmodeldb.info/models/1x-NoiseToner-Poisson-Detailed>

